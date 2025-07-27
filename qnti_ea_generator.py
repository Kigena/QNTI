"""
QNTI EA Generator - Advanced Trading Strategy Generation System
Creates new Expert Advisors from scratch using AI-driven strategy combinations
"""

import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import time
import threading
import queue

# Configure logging
# Import centralized logging
from qnti_logging_utils import get_qnti_logger
logger = get_qnti_logger('QNTI_EAGenerator')

class GenerationStatus(Enum):
    """EA Generation Status"""
    IDLE = "idle"
    GENERATING = "generating"
    TESTING = "testing"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DataSourceConfig:
    """Data source configuration"""
    source: str = "mt5_historical"  # mt5_historical, interactive_brokers, alpaca, yahoo_finance, custom_csv
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    spread: float = 2.0
    commission: float = 5.0
    slippage: float = 1.0

@dataclass
class GeneratorConfig:
    """Generator configuration"""
    generation_time_minutes: int = 60
    max_entry_indicators: int = 3
    max_exit_indicators: int = 2
    max_strategies: int = 1000
    indicator_preset: str = "custom_mix"
    out_of_sample_percent: int = 30
    in_sample_percent: int = 70
    walk_forward_steps: int = 5
    min_trades_per_strategy: int = 50

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    direction: str = "both"  # both, long_only, short_only
    opposite_entry: bool = False
    max_open_trades: int = 1
    lot_sizing_method: str = "fixed"  # fixed, percent_risk, kelly, martingale
    fixed_lot_size: float = 0.1
    risk_percent: float = 2.0
    max_lot_size: float = 10.0
    sl_type: str = "fixed_pips"  # fixed_pips, atr_multiple, percent_balance, dynamic
    sl_value: float = 50.0
    tp_type: str = "fixed_pips"
    tp_value: float = 100.0
    use_trailing_stop: bool = False
    trailing_stop_distance: float = 20.0
    use_break_even: bool = False
    break_even_distance: float = 30.0

@dataclass
class SearchCriteria:
    """Search criteria for filtering strategies"""
    profit_factor: Dict[str, Any] = None
    sharpe_ratio: Dict[str, Any] = None
    max_drawdown: Dict[str, Any] = None
    win_rate: Dict[str, Any] = None
    total_trades: Dict[str, Any] = None
    net_profit: Dict[str, Any] = None

@dataclass
class AdvancedSettings:
    """Advanced generation settings"""
    use_monte_carlo_validation: bool = True
    monte_carlo_runs: int = 1000
    stress_test_enabled: bool = True
    optimize_parameters: bool = True
    eliminate_redundancy: bool = True
    correlation_threshold: float = 0.8
    minimum_robustness_score: float = 0.7

@dataclass
class ComprehensiveConfig:
    """Complete configuration for EA generation"""
    data_source: DataSourceConfig
    generator: GeneratorConfig
    strategy: StrategyConfig
    criteria: SearchCriteria
    advanced: AdvancedSettings

@dataclass
class GeneratedStrategy:
    """Generated strategy result"""
    id: str
    name: str
    status: str = "testing"  # testing, passed, failed
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    net_profit: float = 0.0
    robustness_score: float = 0.0
    entry_rules: int = 1
    exit_rules: int = 1
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    profit_per_month: float = 0.0
    recovery_factor: float = 0.0
    created_at: str = ""
    strategy_code: str = ""
    parameters: Dict[str, Any] = None

class QNTIEAGenerator:
    """Advanced EA Generator System"""
    
    def __init__(self, strategy_tester=None, output_dir: str = "qnti_generated_eas"):
        self.strategy_tester = strategy_tester
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generation state
        self.status = GenerationStatus.IDLE
        self.current_config = None
        self.generated_count = 0
        self.total_to_generate = 0
        self.generation_thread = None
        self.stop_requested = False
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.status_callbacks = []
        self.generated_strategies = []
        self.generation_start_time = None
        
        # Performance tracking
        self.testing_count = 0
        self.passed_count = 0
        self.failed_count = 0
        
        # Strategy templates and indicators
        self.indicator_presets = self._load_indicator_presets()
        self.available_indicators = self._load_available_indicators()
        self.indicator_parameters = self._load_indicator_parameters()
        self.strategy_templates = self._load_strategy_templates()
        
        logger.info("QNTI EA Generator initialized")

    def _load_indicator_presets(self) -> Dict[str, List[str]]:
        """Load comprehensive indicator presets with ALL indicators properly categorized"""
        return {
            "trend_following": [
                "Moving Average", "Exponential Moving Average", "Simple Moving Average", 
                "Weighted Moving Average", "Adaptive Moving Average", "Hull Moving Average",
                "MACD", "Moving Average Convergence Divergence", "ADX", "Average Directional Movement Rating",
                "Directional Movement Indicator", "Plus Directional Indicator", "Minus Directional Indicator",
                "Parabolic SAR", "Ichimoku", "Alligator", "Linear Regression", "Linear Regression Slope",
                "Trend Detection Index", "Qstick", "Vertical Horizontal Filter", "MTF Moving Average",
                "MTF MACD", "MTF ADX", "Zero Lag Exponential Moving Average", "T3 Moving Average",
                "Kaufman Adaptive Moving Average", "Fractal Adaptive Moving Average", "Variable Moving Average",
                "Triangular Moving Average", "Time Series Forecast", "Least Squares Moving Average"
            ],
            "mean_reversion": [
                "RSI", "Relative Strength Index Wilder", "Connors RSI", "Bollinger Bands", "Bollinger Band Width",
                "Bollinger %b", "Stochastic", "Stochastic Fast", "Stochastic Slow", "Full Stochastic",
                "Stochastic RSI", "Stochastic RSI Fast", "Stochastic RSI Slow", "Double Smoothed Stochastic",
                "Williams %R", "CCI", "Chande Momentum Oscillator", "Ultimate Oscillator", "DeMarker",
                "RVI", "OsMA", "MTF RSI", "MTF Stochastic", "MTF CCI", "MTF Williams %R"
            ],
            "momentum": [
                "RSI", "CCI", "Williams %R", "Momentum", "Momentum Oscillator", "Price Momentum Oscillator",
                "Rate of Change", "Price Rate of Change", "Volume Rate of Change", "Relative Momentum Index",
                "Stochastic Momentum Index", "Dynamic Momentum Index", "Chande Momentum", "Chande Momentum Oscillator",
                "Percentage Price Oscillator", "Absolute Price Oscillator", "Relative Strength Comparative",
                "Rainbow Oscillator", "MTF Momentum", "Commodity Selection Index"
            ],
            "volatility": [
                "Bollinger Bands", "Bollinger Band Width", "Bollinger %b", "ATR", "Standard Deviation",
                "Variance", "Keltner Channels", "Donchian Channels", "Price Channels", "Envelopes",
                "Historical Volatility", "Relative Volatility Index", "Intraday Intensity", "Volatility Stop",
                "Volatility Ratio", "Chaikin Volatility", "Standard Error", "Standard Error Bands"
            ],
            "oscillators": [
                "RSI", "Stochastic", "Stochastic Fast", "Stochastic Slow", "CCI", "Williams %R", 
                "Ultimate Oscillator", "Awesome Oscillator", "Accelerator", "Momentum", "RVI", "DeMarker",
                "OsMA", "Aroon Oscillator", "TRIX", "Chande Momentum Oscillator", "Mass Index",
                "Vortex Indicator", "Know Sure Thing", "Pretty Good Oscillator", "Stochastic RSI",
                "Double Smoothed Stochastic", "Stochastic Momentum Index", "Volume Oscillator",
                "Volume Accumulation Oscillator", "Chaikin Oscillator"
            ],
            "price_action": [
                "Price Action", "Candlestick Patterns", "Pin Bar Pattern", "Doji Pattern", "Hammer Pattern",
                "Engulfing Pattern", "Morning Star Pattern", "Evening Star Pattern", "Shooting Star Pattern",
                "Hanging Man Pattern", "Hammer Candlestick", "Shooting Star Candlestick", "Hanging Man Candlestick",
                "Inverted Hammer", "Dragonfly Doji", "Gravestone Doji", "Long Legged Doji", "Spinning Top",
                "Marubozu", "Three White Soldiers", "Three Black Crows", "Rising Three Methods", "Falling Three Methods",
                "Tweezer Top", "Tweezer Bottom", "Dark Cloud Cover", "Piercing Pattern", "Bullish Harami",
                "Bearish Harami", "Harami Cross", "Market Profile", "Point and Figure", "Renko", "Three Line Break",
                "Heikin Ashi", "Median Price", "Typical Price", "Weighted Close", "Average Price"
            ],
            "volume": [
                "Volumes", "On Balance Volume", "Volume Weighted Average Price", "VWAP", "Chaikin Money Flow",
                "Volume Oscillator", "Price Volume Trend", "Negative Volume Index", "Positive Volume Index",
                "Volume Rate of Change", "Accumulation Distribution Line", "Accumulation/Distribution",
                "Williams Accumulation/Distribution", "Williams Accumulation Distribution", "Ease of Movement",
                "Volume Weighted Moving Average", "Volume Accumulation Oscillator", "Money Flow Index",
                "Market Facilitation Index", "Force Index", "MTF Force Index"
            ],
            "custom_mix": [
                # All standard MT4/MT5 indicators
                "Moving Average", "RSI", "MACD", "Bollinger Bands", "Stochastic", "CCI", "ADX", "ATR",
                "Custom Indicator", "Parabolic SAR", "Momentum", "DeMarker", "RVI", "Williams %R",
                "Accelerator", "Awesome Oscillator", "Fractals", "Gator", "Ichimoku", "Market Facilitation Index",
                "Envelopes", "Force Index", "OsMA", "Standard Deviation", "Variance", "Volumes",
                "Bears Power", "Bulls Power", "Money Flow Index", "On Balance Volume", "Rate of Change",
                "Triple Exponential Moving Average", "Variable Index Dynamic Average", "Williams Accumulation/Distribution",
                "Alligator", "Exponential Moving Average", "Simple Moving Average", "Weighted Moving Average",
                
                # Advanced indicators selection (most commonly used)
                "Linear Regression", "ZigZag", "Fibonacci Retracement", "Pivot Points", "Support/Resistance",
                "VWAP", "Chaikin Oscillator", "Keltner Channels", "Donchian Channels", "Aroon Oscillator",
                "TRIX", "Ultimate Oscillator", "Volume Weighted Average Price", "Chaikin Money Flow",
                "Price Action", "Pin Bar Pattern", "Doji Pattern", "Hammer Pattern", "Engulfing Pattern",
                
                # Modern adaptive indicators
                "Hull Moving Average", "Adaptive Moving Average", "Kaufman Adaptive Moving Average",
                "Zero Lag Exponential Moving Average", "T3 Moving Average", "Stochastic RSI", "Connors RSI",
                "Relative Volatility Index", "Dynamic Momentum Index", "Currency Strength Meter"
            ],
            "fibonacci_analysis": [
                "Fibonacci Retracement", "Fibonacci Extensions", "Fibonacci Fans", "Fibonacci Time Zones",
                "ZigZag", "Support/Resistance", "Pivot Points", "Linear Regression", "Trend Lines"
            ],
            "elliott_wave": [
                "Elliott Wave", "ZigZag", "Fractals", "Fibonacci Retracement", "Fibonacci Extensions",
                "Support/Resistance", "Trend Lines", "Pivot Points", "Wave Analysis"
            ],
            "gann_analysis": [
                "Gann Angles", "Gann Square", "Support/Resistance", "Pivot Points", "Fibonacci Retracement",
                "Time Cycles", "Price Cycles", "Trend Lines"
            ],
            "multi_timeframe": [
                "MTF Moving Average", "MTF RSI", "MTF MACD", "MTF Stochastic", "MTF Bollinger Bands",
                "MTF ADX", "MTF CCI", "MTF Williams %R", "MTF Momentum", "MTF Force Index"
            ],
            "synthetic_analysis": [
                "Currency Strength Meter", "Market Sentiment", "Fear and Greed Index", "Volatility Index",
                "Risk Appetite", "Correlation Coefficient", "Beta Coefficient", "Alpha Ratio",
                "Sharpe Ratio Indicator", "Sortino Ratio Indicator"
            ],
            "advanced_mathematics": [
                "Fourier Transform", "Wavelet Transform", "Hilbert Transform", "Dominant Cycle Period",
                "Instantaneous Trendline", "Trend vs Cycle Mode", "Cycle Analysis", "Linear Regression",
                "Linear Regression Slope", "Linear Regression Intercept", "Standard Error", "Moving Linear Regression"
            ]
        }

    def _load_available_indicators(self) -> List[str]:
        """Load ALL 97+ available indicators from original EA parser system"""
        return [
            # Standard MT4/MT5 indicators (from indicators_map)
            'Moving Average',
            'RSI',
            'MACD',
            'Bollinger Bands',
            'Stochastic',
            'CCI',
            'ADX',
            'ATR',
            'Custom Indicator',
            'Parabolic SAR',
            'Momentum',
            'DeMarker',
            'RVI',
            'Williams %R',
            'Accelerator',
            'Awesome Oscillator',
            'Fractals',
            'Gator',
            'Ichimoku',
            'Market Facilitation Index',
            'Envelopes',
            'Force Index',
            'OsMA',
            'Standard Deviation',
            'Variance',
            'Volumes',
            'Bears Power',
            'Bulls Power',
            'Money Flow Index',
            'On Balance Volume',
            'Rate of Change',
            'Triple Exponential Moving Average',
            'Variable Index Dynamic Average',
            'Williams Accumulation/Distribution',
            'Alligator',
            
            # Additional common patterns
            'Exponential Moving Average',
            'Simple Moving Average',
            'Weighted Moving Average',
            'Linear Weighted Moving Average',
            'Smoothed Moving Average',
            
            # Advanced indicators (from additional_patterns)
            'Detrended Price Oscillator',
            'Linear Regression',
            'ZigZag',
            'Fibonacci Retracement',
            'Pivot Points',
            'Support/Resistance',
            'Trend Lines',
            'VWAP',
            'Accumulation/Distribution',
            'Chaikin Oscillator',
            'Elder Ray',
            'Keltner Channels',
            'Price Channels',
            'Donchian Channels',
            'Aroon Oscillator',
            'TRIX',
            'Ultimate Oscillator',
            'Chande Momentum Oscillator',
            'Mass Index',
            'Vortex Indicator',
            'Know Sure Thing',
            'Pretty Good Oscillator',
            'Schiff Pitchfork',
            'Andrews Pitchfork',
            
            # Price Action & Candlestick Patterns
            'Price Action',
            'Candlestick Patterns',
            'Pin Bar Pattern',
            'Doji Pattern',
            'Hammer Pattern',
            'Engulfing Pattern',
            'Morning Star Pattern',
            'Evening Star Pattern',
            'Shooting Star Pattern',
            'Hanging Man Pattern',
            
            # Volume indicators
            'Volume Weighted Average Price',
            'Chaikin Money Flow',
            'Volume Oscillator',
            'Price Volume Trend',
            'Negative Volume Index',
            'Positive Volume Index',
            'Volume Rate of Change',
            'Accumulation Distribution Line',
            'Ease of Movement',
            'Volume Weighted Moving Average',
            
            # Volatility indicators
            'Bollinger Band Width',
            'Bollinger %b',
            'Historical Volatility',
            'Relative Volatility Index',
            'Intraday Intensity',
            'Volatility Stop',
            'Volatility Ratio',
            'Chaikin Volatility',
            
            # Momentum indicators
            'Relative Momentum Index',
            'Stochastic RSI',
            'Double Smoothed Stochastic',
            'Stochastic Momentum Index',
            'Commodity Selection Index',
            'Relative Strength Comparative',
            'Price Rate of Change',
            'Momentum Oscillator',
            'Percentage Price Oscillator',
            'Absolute Price Oscillator',
            
            # Trend indicators
            'Average Directional Movement Rating',
            'Directional Movement Indicator',
            'Plus Directional Indicator',
            'Minus Directional Indicator',
            'Trend Detection Index',
            'Qstick',
            'Vertical Horizontal Filter',
            'Moving Average Convergence Divergence',
            'Price Momentum Oscillator',
            'Rainbow Oscillator',
            
            # Custom and specialized indicators
            'Adaptive Moving Average',
            'Kaufman Adaptive Moving Average',
            'Zero Lag Exponential Moving Average',
            'Hull Moving Average',
            'Jurik Moving Average',
            'T3 Moving Average',
            'Fractal Adaptive Moving Average',
            'Variable Moving Average',
            'Triangular Moving Average',
            'Time Series Forecast',
            'Linear Regression Slope',
            'Linear Regression Intercept',
            'Standard Error',
            'Standard Error Bands',
            'Least Squares Moving Average',
            'Moving Linear Regression',
            
            # Oscillators
            'Stochastic Fast',
            'Stochastic Slow',
            'Full Stochastic',
            'Slow Stochastic',
            'Fast Stochastic',
            'Williams Accumulation Distribution',
            'Volume Accumulation Oscillator',
            'Chande Momentum',
            'Dynamic Momentum Index',
            'Relative Strength Index Wilder',
            'Connors RSI',
            'Stochastic RSI Fast',
            'Stochastic RSI Slow',
            
            # Market structure indicators
            'Market Profile',
            'Point and Figure',
            'Renko',
            'Three Line Break',
            'Heikin Ashi',
            'Median Price',
            'Typical Price',
            'Weighted Close',
            'Average Price',
            
            # Pattern recognition
            'Hammer Candlestick',
            'Shooting Star Candlestick',
            'Hanging Man Candlestick',
            'Inverted Hammer',
            'Dragonfly Doji',
            'Gravestone Doji',
            'Long Legged Doji',
            'Spinning Top',
            'Marubozu',
            'Three White Soldiers',
            'Three Black Crows',
            'Rising Three Methods',
            'Falling Three Methods',
            'Tweezer Top',
            'Tweezer Bottom',
            'Dark Cloud Cover',
            'Piercing Pattern',
            'Bullish Harami',
            'Bearish Harami',
            'Harami Cross',
            
            # Advanced analysis
            'Elliott Wave',
            'Fibonacci Extensions',
            'Fibonacci Fans',
            'Fibonacci Time Zones',
            'Gann Angles',
            'Gann Square',
            'Cycle Analysis',
            'Fourier Transform',
            'Wavelet Transform',
            'Hilbert Transform',
            'Dominant Cycle Period',
            'Instantaneous Trendline',
            'Trend vs Cycle Mode',
            
            # Multi-timeframe indicators
            'MTF Moving Average',
            'MTF RSI',
            'MTF MACD',
            'MTF Stochastic',
            'MTF Bollinger Bands',
            'MTF ADX',
            'MTF CCI',
            'MTF Williams %R',
            'MTF Momentum',
            'MTF Force Index',
            
            # Synthetic indicators
            'Currency Strength Meter',
            'Market Sentiment',
            'Fear and Greed Index',
            'Volatility Index',
            'Risk Appetite',
            'Correlation Coefficient',
            'Beta Coefficient',
            'Alpha Ratio',
            'Sharpe Ratio Indicator',
            'Sortino Ratio Indicator'
        ]

    def _load_strategy_templates(self) -> Dict[str, Dict]:
        """Load comprehensive strategy templates"""
        return {
            "trend_following": {
                "name": "Advanced Trend Following",
                "description": "Multi-timeframe trend analysis with momentum confirmation",
                "indicators": ["EMA", "MACD", "ADX"],
                "entry_logic": "ema_trend_and_macd_signal_and_adx_strong",
                "exit_logic": "ema_trend_change_or_macd_divergence",
                "risk_reward": 2.0,
                "complexity": "medium"
            },
            "mean_reversion": {
                "name": "Smart Mean Reversion",
                "description": "RSI and Bollinger Bands with volume confirmation",
                "indicators": ["RSI", "Bollinger_Bands", "Volume"],
                "entry_logic": "rsi_extreme_and_bb_touch_and_volume_spike",
                "exit_logic": "rsi_normalize_or_bb_opposite_band",
                "risk_reward": 1.5,
                "complexity": "medium"
            },
            "breakout": {
                "name": "Volatility Breakout",
                "description": "High-probability breakout detection with volatility filter",
                "indicators": ["ATR", "Bollinger_Bands", "Volume"],
                "entry_logic": "bb_squeeze_break_with_volume_and_atr",
                "exit_logic": "volatility_decrease_or_bb_return",
                "risk_reward": 2.5,
                "complexity": "high"
            },
            "scalping": {
                "name": "Precision Scalping",
                "description": "Quick profits with tight risk management",
                "indicators": ["EMA", "Stochastic", "RSI"],
                "entry_logic": "ema_alignment_and_stoch_cross_and_rsi_confirm",
                "exit_logic": "quick_profit_or_momentum_loss",
                "risk_reward": 1.0,
                "complexity": "low"
            },
            "swing_trading": {
                "name": "Swing Momentum",
                "description": "Multi-day swings with trend confirmation",
                "indicators": ["SMA", "MACD", "ADX"],
                "entry_logic": "sma_trend_and_macd_histogram_and_adx_rising",
                "exit_logic": "trend_exhaustion_or_time_based",
                "risk_reward": 3.0,
                "complexity": "medium"
            },
            "grid_trading": {
                "name": "Adaptive Grid",
                "description": "Dynamic grid spacing based on volatility",
                "indicators": ["ATR", "SMA", "RSI"],
                "entry_logic": "grid_level_with_trend_and_momentum",
                "exit_logic": "grid_profit_or_trend_reversal",
                "risk_reward": 1.2,
                "complexity": "high"
            }
        }

    def start_comprehensive_generation(self, config: ComprehensiveConfig) -> Dict[str, Any]:
        """Start comprehensive EA generation with full configuration"""
        if self.status != GenerationStatus.IDLE:
            return {"status": "error", "message": "Generation already in progress"}

        try:
            self.current_config = config
            self.status = GenerationStatus.GENERATING
            self.generated_count = 0
            self.total_to_generate = config.generator.max_strategies
            self.stop_requested = False
            self.generated_strategies = []
            self.generation_start_time = time.time()
            
            # Reset counters
            self.testing_count = 0
            self.passed_count = 0
            self.failed_count = 0

            # Start generation thread
            self.generation_thread = threading.Thread(
                target=self._comprehensive_generation_thread,
                args=(config,)
            )
            self.generation_thread.start()

            logger.info(f"Started comprehensive EA generation with {config.generator.max_strategies} target strategies")
            return {"status": "started", "message": "EA generation started successfully"}

        except Exception as e:
            logger.error(f"Failed to start EA generation: {str(e)}")
            self.status = GenerationStatus.FAILED
            return {"status": "error", "message": str(e)}

    def _comprehensive_generation_thread(self, config: ComprehensiveConfig):
        """Main generation thread with comprehensive logic"""
        try:
            logger.info("🚀 Starting comprehensive EA generation...")
            
            generation_time_limit = config.generator.generation_time_minutes * 60
            start_time = time.time()
            
            # Get indicators for this preset
            indicators = self.indicator_presets.get(
                config.generator.indicator_preset, 
                self.indicator_presets["custom_mix"]
            )
            
            while (time.time() - start_time < generation_time_limit and 
                   self.generated_count < config.generator.max_strategies and 
                   not self.stop_requested):
                
                try:
                    # Generate a new strategy
                    strategy = self._generate_comprehensive_strategy(config, indicators)
                    
                    if strategy:
                        # Test the strategy
                        self.testing_count += 1
                        self._update_progress(f"Testing strategy {strategy.name}...")
                        
                        # Run validation
                        if self._validate_strategy(strategy, config):
                            strategy.status = "passed"
                            self.passed_count += 1
                            logger.info(f"✅ Strategy {strategy.name} passed validation")
                        else:
                            strategy.status = "failed"
                            self.failed_count += 1
                            logger.info(f"❌ Strategy {strategy.name} failed validation")
                        
                        # Add to results
                        self.generated_strategies.append(strategy)
                        self.generated_count += 1
                        
                        # Notify callbacks
                        for callback in self.status_callbacks:
                            try:
                                callback(self._get_progress_data())
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        # Brief pause between generations
                        time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Error generating strategy: {e}")
                    continue
            
            # Generation complete
            self.status = GenerationStatus.COMPLETED
            generation_time = time.time() - start_time
            
            logger.info(f"🎉 Generation completed in {generation_time:.2f}s")
            logger.info(f"📊 Generated: {self.generated_count}, Passed: {self.passed_count}, Failed: {self.failed_count}")
            
            self._update_progress("✅ Generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Generation thread error: {e}")
            self.status = GenerationStatus.FAILED
            self._update_progress(f"❌ Generation failed: {str(e)}")

    def _generate_comprehensive_strategy(self, config: ComprehensiveConfig, available_indicators: List[str]) -> Optional[GeneratedStrategy]:
        """Generate a comprehensive strategy with all configuration options"""
        try:
            # Generate unique strategy
            strategy_id = str(uuid.uuid4())
            strategy_name = f"Strategy_{int(time.time() * 1000) % 10000}"
            
            # Select random template
            template_name = random.choice(list(self.strategy_templates.keys()))
            template = self.strategy_templates[template_name]
            
            # Generate indicators (respecting max limits)
            max_entry = config.generator.max_entry_indicators
            max_exit = config.generator.max_exit_indicators
            
            entry_indicators = random.sample(
                available_indicators, 
                min(max_entry, len(available_indicators))
            )
            exit_indicators = random.sample(
                available_indicators,
                min(max_exit, len(available_indicators))
            )
            
            # Create strategy
            strategy = GeneratedStrategy(
                id=strategy_id,
                name=strategy_name,
                status="testing",
                entry_rules=len(entry_indicators),
                exit_rules=len(exit_indicators),
                created_at=datetime.now().isoformat(),
                parameters={
                    "template": template_name,
                    "entry_indicators": entry_indicators,
                    "exit_indicators": exit_indicators,
                    "data_source": asdict(config.data_source),
                    "strategy_config": asdict(config.strategy)
                }
            )
            
            # Generate performance metrics (simulated for now)
            strategy = self._simulate_strategy_performance(strategy, config)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return None

    def _simulate_strategy_performance(self, strategy: GeneratedStrategy, config: ComprehensiveConfig) -> GeneratedStrategy:
        """Simulate strategy performance metrics"""
        try:
            # Generate realistic performance metrics
            base_profit_factor = 1.0 + random.random() * 2.5  # 1.0 to 3.5
            base_sharpe = 0.5 + random.random() * 2.0  # 0.5 to 2.5
            base_drawdown = 5 + random.random() * 35  # 5% to 40%
            base_win_rate = 30 + random.random() * 50  # 30% to 80%
            
            # Adjust based on complexity and template
            template = self.strategy_templates.get(strategy.parameters["template"], {})
            complexity_factor = {"low": 0.8, "medium": 1.0, "high": 1.2}.get(
                template.get("complexity", "medium"), 1.0
            )
            
            strategy.profit_factor = round(base_profit_factor * complexity_factor, 2)
            strategy.sharpe_ratio = round(base_sharpe * complexity_factor, 2)
            strategy.max_drawdown = round(base_drawdown / complexity_factor, 1)
            strategy.win_rate = round(base_win_rate, 1)
            strategy.total_trades = random.randint(50, 500)
            strategy.net_profit = round(10 + random.random() * 150, 1)
            strategy.robustness_score = round(0.3 + random.random() * 0.6, 2)
            
            # Additional metrics
            strategy.avg_win = round(50 + random.random() * 150, 2)
            strategy.avg_loss = round(20 + random.random() * 80, 2)
            strategy.max_consecutive_wins = random.randint(3, 15)
            strategy.max_consecutive_losses = random.randint(2, 8)
            strategy.profit_per_month = round(1 + random.random() * 15, 2)
            strategy.recovery_factor = round(1.5 + random.random() * 3.5, 2)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error simulating performance: {e}")
            return strategy

    def _validate_strategy(self, strategy: GeneratedStrategy, config: ComprehensiveConfig) -> bool:
        """Validate strategy against search criteria"""
        try:
            criteria = config.criteria
            
            # Check profit factor
            if (criteria.profit_factor and 
                criteria.profit_factor.get("enabled", True) and
                strategy.profit_factor < criteria.profit_factor.get("value", 1.5)):
                return False
            
            # Check Sharpe ratio
            if (criteria.sharpe_ratio and 
                criteria.sharpe_ratio.get("enabled", True) and
                strategy.sharpe_ratio < criteria.sharpe_ratio.get("value", 1.0)):
                return False
            
            # Check max drawdown
            if (criteria.max_drawdown and 
                criteria.max_drawdown.get("enabled", True) and
                strategy.max_drawdown > criteria.max_drawdown.get("value", 20)):
                return False
            
            # Check win rate
            if (criteria.win_rate and 
                criteria.win_rate.get("enabled", True) and
                strategy.win_rate < criteria.win_rate.get("value", 50)):
                return False
            
            # Check total trades
            if (criteria.total_trades and 
                criteria.total_trades.get("enabled", True) and
                strategy.total_trades < criteria.total_trades.get("value", 100)):
                return False
            
            # Check net profit
            if (criteria.net_profit and 
                criteria.net_profit.get("enabled", True) and
                strategy.net_profit < criteria.net_profit.get("value", 20)):
                return False
            
            # Check robustness score
            if (strategy.robustness_score < config.advanced.minimum_robustness_score):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating strategy: {e}")
            return False

    def _get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data"""
        return {
            "testing": self.testing_count - self.passed_count - self.failed_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total_generated": self.generated_count,
            "avgProfitFactor": self._calculate_avg_profit_factor(),
            "bestSharpe": self._calculate_best_sharpe(),
            "latest_strategy": self.generated_strategies[-1] if self.generated_strategies else None
        }

    def _calculate_avg_profit_factor(self) -> float:
        """Calculate average profit factor of generated strategies"""
        if not self.generated_strategies:
            return 0.0
        return round(sum(s.profit_factor for s in self.generated_strategies) / len(self.generated_strategies), 2)

    def _calculate_best_sharpe(self) -> float:
        """Calculate best Sharpe ratio of generated strategies"""
        if not self.generated_strategies:
            return 0.0
        return round(max(s.sharpe_ratio for s in self.generated_strategies), 2)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive generation status"""
        status_data = {
            "status": self.status.value,
            "generated_count": self.generated_count,
            "total_to_generate": self.total_to_generate,
            "testing_count": self.testing_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "progress_percentage": (self.generated_count / max(self.total_to_generate, 1)) * 100,
            "strategies": [asdict(strategy) for strategy in self.generated_strategies[-20:]],  # Last 20
            "performance_stats": {
                "avg_profit_factor": self._calculate_avg_profit_factor(),
                "best_sharpe": self._calculate_best_sharpe(),
                "success_rate": (self.passed_count / max(self.generated_count, 1)) * 100 if self.generated_count > 0 else 0
            }
        }
        
        if self.generation_start_time:
            status_data["elapsed_time"] = time.time() - self.generation_start_time
        
        return status_data

    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all generated strategies"""
        return [asdict(strategy) for strategy in self.generated_strategies]

    def stop_generation(self):
        """Stop EA generation"""
        logger.info("⏹️ Stopping EA generation...")
        self.stop_requested = True
        self.status = GenerationStatus.IDLE
        self._update_progress("⏹️ Generation stopped by user")

    def _update_progress(self, message: str):
        """Update progress with message"""
        progress_data = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "generated_count": self.generated_count,
            "status": self.status.value
        }
        
        try:
            self.progress_queue.put_nowait(progress_data)
        except queue.Full:
            pass  # Skip if queue is full
        
        logger.info(f"Progress: {message}")

    def add_status_callback(self, callback):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators"""
        return self.available_indicators.copy()

    def get_indicator_presets(self) -> Dict[str, List[str]]:
        """Get indicator presets"""
        return self.indicator_presets.copy()

    def export_strategy_to_mql5(self, strategy_id: str) -> Optional[str]:
        """Export strategy to MQL5 code"""
        strategy = next((s for s in self.generated_strategies if s.id == strategy_id), None)
        if not strategy:
            return None
        
        # Generate basic MQL5 code structure
        mql5_code = f"""
//+------------------------------------------------------------------+
//|                                               {strategy.name}.mq5 |
//|                                  Generated by QNTI EA Generator |
//+------------------------------------------------------------------+

#property copyright "QNTI Trading System"
#property version   "1.00"
#property indicator_chart_window

// Strategy Parameters
input double LotSize = {strategy.parameters.get('lot_size', 0.1)};
input int StopLoss = {strategy.parameters.get('sl_value', 50)};
input int TakeProfit = {strategy.parameters.get('tp_value', 100)};
input int MagicNumber = {random.randint(10000, 99999)};

// Global variables
int ticket = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {{
   Print("EA {strategy.name} initialized");
   return(INIT_SUCCEEDED);
  }}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {{
   // Entry logic based on {strategy.parameters.get('template', 'unknown')} template
   if(CheckEntryConditions())
     {{
      if(OrdersTotal() == 0)
        {{
         // Open trade logic here
         OpenTrade();
        }}
     }}
   
   // Exit logic
   CheckExitConditions();
  }}

//+------------------------------------------------------------------+
//| Check entry conditions                                           |
//+------------------------------------------------------------------+
bool CheckEntryConditions()
  {{
   // Implementation of entry conditions
   // Based on indicators: {', '.join(strategy.parameters.get('entry_indicators', []))}
   return true; // Placeholder
  }}

//+------------------------------------------------------------------+
//| Open trade                                                       |
//+------------------------------------------------------------------+
void OpenTrade()
  {{
   // Implementation of trade opening logic
   Print("Opening trade for {strategy.name}");
  }}

//+------------------------------------------------------------------+
//| Check exit conditions                                            |
//+------------------------------------------------------------------+
void CheckExitConditions()
  {{
   // Implementation of exit conditions
   // Based on indicators: {', '.join(strategy.parameters.get('exit_indicators', []))}
  }}
"""
        
        return mql5_code

    def _load_indicator_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive indicator parameters with granular configuration for each indicator"""
        return {
            # MOVING AVERAGES
            "Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "method": {"type": "select", "options": ["Simple", "Exponential", "Smoothed", "Linear Weighted"], "default": "Simple"},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Moving Average with configurable calculation method"
            },
            "Exponential Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 21, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Exponential Moving Average with exponential smoothing"
            },
            "Simple Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 20, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Simple Moving Average - arithmetic mean of prices"
            },
            "Weighted Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "weight_method": {"type": "select", "options": ["Linear", "Exponential", "Triangular"], "default": "Linear"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Weighted Moving Average with configurable weighting"
            },
            "Hull Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 16, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "smoothing_factor": {"type": "float", "min": 0.1, "max": 3.0, "default": 2.0, "step": 0.1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Hull Moving Average - reduced lag moving average"
            },
            "Adaptive Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "fast_period": {"type": "int", "min": 1, "max": 50, "default": 2, "step": 1},
                    "slow_period": {"type": "int", "min": 1, "max": 200, "default": 30, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Adaptive Moving Average - adjusts to market volatility"
            },

            # OSCILLATORS
            "RSI": {
                "parameters": {
                    "period": {"type": "int", "min": 2, "max": 200, "default": 14, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "overbought_level": {"type": "float", "min": 50, "max": 95, "default": 70, "step": 1},
                    "oversold_level": {"type": "float", "min": 5, "max": 50, "default": 30, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Momentum",
                "description": "Relative Strength Index - momentum oscillator"
            },
            "Stochastic": {
                "parameters": {
                    "k_period": {"type": "int", "min": 1, "max": 200, "default": 5, "step": 1},
                    "d_period": {"type": "int", "min": 1, "max": 200, "default": 3, "step": 1},
                    "slowing": {"type": "int", "min": 1, "max": 50, "default": 3, "step": 1},
                    "ma_method": {"type": "select", "options": ["Simple", "Exponential", "Smoothed", "Linear Weighted"], "default": "Simple"},
                    "price_field": {"type": "select", "options": ["Low/High", "Close/Close"], "default": "Low/High"},
                    "overbought_level": {"type": "float", "min": 50, "max": 95, "default": 80, "step": 1},
                    "oversold_level": {"type": "float", "min": 5, "max": 50, "default": 20, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Momentum",
                "description": "Stochastic Oscillator - momentum indicator"
            },
            "Williams %R": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "overbought_level": {"type": "float", "min": -50, "max": -5, "default": -20, "step": 1},
                    "oversold_level": {"type": "float", "min": -95, "max": -50, "default": -80, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Momentum",
                "description": "Williams Percent Range - momentum oscillator"
            },
            "CCI": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "applied_to": {"type": "select", "options": ["Typical", "Close", "Weighted", "Median"], "default": "Typical"},
                    "overbought_level": {"type": "float", "min": 50, "max": 300, "default": 100, "step": 10},
                    "oversold_level": {"type": "float", "min": -300, "max": -50, "default": -100, "step": 10},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Momentum",
                "description": "Commodity Channel Index - momentum oscillator"
            },

            # MACD
            "MACD": {
                "parameters": {
                    "fast_ema": {"type": "int", "min": 1, "max": 200, "default": 12, "step": 1},
                    "slow_ema": {"type": "int", "min": 1, "max": 200, "default": 26, "step": 1},
                    "signal_sma": {"type": "int", "min": 1, "max": 200, "default": 9, "step": 1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Moving Average Convergence Divergence - trend following momentum indicator"
            },

            # BOLLINGER BANDS
            "Bollinger Bands": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 20, "step": 1},
                    "deviation": {"type": "float", "min": 0.1, "max": 5.0, "default": 2.0, "step": 0.1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volatility",
                "description": "Bollinger Bands - volatility indicator with standard deviation bands"
            },
            "Bollinger Band Width": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 20, "step": 1},
                    "deviation": {"type": "float", "min": 0.1, "max": 5.0, "default": 2.0, "step": 0.1},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "normalization": {"type": "select", "options": ["Percentage", "Absolute", "Z-Score"], "default": "Percentage"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volatility",
                "description": "Bollinger Band Width - measures band width for volatility analysis"
            },

            # ATR
            "ATR": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volatility",
                "description": "Average True Range - volatility indicator"
            },

            # ADX
            "ADX": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "applied_to": {"type": "select", "options": ["High/Low", "Close"], "default": "High/Low"},
                    "trend_threshold": {"type": "float", "min": 10, "max": 50, "default": 25, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Average Directional Index - trend strength indicator"
            },

            # PARABOLIC SAR
            "Parabolic SAR": {
                "parameters": {
                    "step": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.02, "step": 0.001},
                    "maximum": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.2, "step": 0.01},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Parabolic Stop and Reverse - trend following indicator"
            },

            # ICHIMOKU
            "Ichimoku": {
                "parameters": {
                    "tenkan_sen": {"type": "int", "min": 1, "max": 100, "default": 9, "step": 1},
                    "kijun_sen": {"type": "int", "min": 1, "max": 100, "default": 26, "step": 1},
                    "senkou_span_b": {"type": "int", "min": 1, "max": 200, "default": 52, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Trend",
                "description": "Ichimoku Kinko Hyo - comprehensive trend and momentum system"
            },

            # VOLUME INDICATORS
            "On Balance Volume": {
                "parameters": {
                    "applied_to": {"type": "select", "options": ["Close", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volume",
                "description": "On Balance Volume - volume momentum indicator"
            },
            "Volume Weighted Average Price": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 500, "default": 14, "step": 1},
                    "price_type": {"type": "select", "options": ["Typical", "Close", "Weighted", "High", "Low"], "default": "Typical"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volume",
                "description": "Volume Weighted Average Price - volume and price analysis"
            },
            "Money Flow Index": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 14, "step": 1},
                    "overbought_level": {"type": "float", "min": 50, "max": 95, "default": 80, "step": 1},
                    "oversold_level": {"type": "float", "min": 5, "max": 50, "default": 20, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Volume",
                "description": "Money Flow Index - volume-weighted RSI"
            },

            # FIBONACCI
            "Fibonacci Retracement": {
                "parameters": {
                    "high_price": {"type": "float", "min": 0, "max": 100000, "default": 0, "step": 0.00001},
                    "low_price": {"type": "float", "min": 0, "max": 100000, "default": 0, "step": 0.00001},
                    "fib_levels": {"type": "multi_select", "options": ["0.236", "0.382", "0.500", "0.618", "0.786"], "default": ["0.382", "0.618"]},
                    "extend_lines": {"type": "boolean", "default": True},
                    "show_prices": {"type": "boolean", "default": True}
                },
                "category": "Support/Resistance",
                "description": "Fibonacci Retracement - key support and resistance levels"
            },

            # PATTERN RECOGNITION
            "Pin Bar Pattern": {
                "parameters": {
                    "body_ratio": {"type": "float", "min": 0.1, "max": 0.5, "default": 0.3, "step": 0.1},
                    "wick_ratio": {"type": "float", "min": 1.5, "max": 5.0, "default": 2.0, "step": 0.1},
                    "min_size_pips": {"type": "int", "min": 5, "max": 100, "default": 10, "step": 1},
                    "confirmation_candles": {"type": "int", "min": 0, "max": 5, "default": 1, "step": 1}
                },
                "category": "Price Action",
                "description": "Pin Bar Pattern - reversal candlestick pattern"
            },
            "Doji Pattern": {
                "parameters": {
                    "body_threshold": {"type": "float", "min": 0.01, "max": 0.2, "default": 0.05, "step": 0.01},
                    "wick_symmetry": {"type": "float", "min": 0.5, "max": 2.0, "default": 1.0, "step": 0.1},
                    "min_total_range": {"type": "int", "min": 5, "max": 100, "default": 15, "step": 1},
                    "confirmation_required": {"type": "boolean", "default": True}
                },
                "category": "Price Action",
                "description": "Doji Pattern - indecision candlestick pattern"
            },

            # ADVANCED INDICATORS
            "Currency Strength Meter": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 28, "step": 1},
                    "smoothing": {"type": "int", "min": 1, "max": 50, "default": 3, "step": 1},
                    "currencies": {"type": "multi_select", "options": ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"], "default": ["USD", "EUR", "GBP", "JPY"]},
                    "calculation_method": {"type": "select", "options": ["RSI", "Linear Regression", "Momentum"], "default": "RSI"}
                },
                "category": "Synthetic",
                "description": "Currency Strength Meter - relative currency strength analysis"
            },
            "Market Sentiment": {
                "parameters": {
                    "sentiment_period": {"type": "int", "min": 1, "max": 100, "default": 14, "step": 1},
                    "volatility_weight": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0, "step": 0.1},
                    "volume_weight": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0, "step": 0.1},
                    "price_weight": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0, "step": 0.1},
                    "normalization_period": {"type": "int", "min": 10, "max": 500, "default": 100, "step": 1}
                },
                "category": "Synthetic",
                "description": "Market Sentiment - combined market mood indicator"
            },

            # MATHEMATICAL TRANSFORMS
            "Fourier Transform": {
                "parameters": {
                    "period": {"type": "int", "min": 8, "max": 256, "default": 64, "step": 8},
                    "window_function": {"type": "select", "options": ["Rectangular", "Hamming", "Hanning", "Blackman"], "default": "Hamming"},
                    "harmonics_count": {"type": "int", "min": 1, "max": 20, "default": 5, "step": 1},
                    "frequency_cutoff": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01}
                },
                "category": "Mathematical",
                "description": "Fourier Transform - frequency domain analysis"
            },
            "Hilbert Transform": {
                "parameters": {
                    "period": {"type": "int", "min": 4, "max": 200, "default": 20, "step": 1},
                    "phase_threshold": {"type": "float", "min": 0.1, "max": 3.14, "default": 1.57, "step": 0.1},
                    "amplitude_threshold": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
                    "smoothing_factor": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.7, "step": 0.1}
                },
                "category": "Mathematical",
                "description": "Hilbert Transform - instantaneous phase and amplitude"
            },

            # MULTI-TIMEFRAME
            "MTF Moving Average": {
                "parameters": {
                    "period": {"type": "int", "min": 1, "max": 200, "default": 20, "step": 1},
                    "timeframe": {"type": "select", "options": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"], "default": "H1"},
                    "method": {"type": "select", "options": ["Simple", "Exponential", "Smoothed", "Linear Weighted"], "default": "Simple"},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Multi-Timeframe",
                "description": "Multi-Timeframe Moving Average - higher timeframe trend analysis"
            },
            "MTF RSI": {
                "parameters": {
                    "period": {"type": "int", "min": 2, "max": 200, "default": 14, "step": 1},
                    "timeframe": {"type": "select", "options": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"], "default": "H4"},
                    "applied_to": {"type": "select", "options": ["Close", "Open", "High", "Low", "Median", "Typical", "Weighted"], "default": "Close"},
                    "overbought_level": {"type": "float", "min": 50, "max": 95, "default": 70, "step": 1},
                    "oversold_level": {"type": "float", "min": 5, "max": 50, "default": 30, "step": 1},
                    "shift": {"type": "int", "min": 0, "max": 50, "default": 0, "step": 1}
                },
                "category": "Multi-Timeframe",
                "description": "Multi-Timeframe RSI - higher timeframe momentum analysis"
            }
        }

    def get_indicator_parameters(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed parameters for a specific indicator"""
        return self.indicator_parameters.get(indicator_name)

    def get_all_indicator_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all indicator parameters with granular configuration"""
        return self.indicator_parameters.copy()

    def validate_indicator_config(self, indicator_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate indicator configuration against parameter constraints"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        indicator_params = self.get_indicator_parameters(indicator_name)
        if not indicator_params:
            result["valid"] = False
            result["errors"].append(f"Unknown indicator: {indicator_name}")
            return result
        
        parameters = indicator_params.get("parameters", {})
        
        for param_name, param_config in parameters.items():
            if param_name in config:
                value = config[param_name]
                param_type = param_config.get("type")
                
                # Validate based on parameter type
                if param_type == "int":
                    if not isinstance(value, int):
                        result["errors"].append(f"{param_name} must be an integer")
                        result["valid"] = False
                    elif "min" in param_config and value < param_config["min"]:
                        result["errors"].append(f"{param_name} must be >= {param_config['min']}")
                        result["valid"] = False
                    elif "max" in param_config and value > param_config["max"]:
                        result["errors"].append(f"{param_name} must be <= {param_config['max']}")
                        result["valid"] = False
                        
                elif param_type == "float":
                    if not isinstance(value, (int, float)):
                        result["errors"].append(f"{param_name} must be a number")
                        result["valid"] = False
                    elif "min" in param_config and value < param_config["min"]:
                        result["errors"].append(f"{param_name} must be >= {param_config['min']}")
                        result["valid"] = False
                    elif "max" in param_config and value > param_config["max"]:
                        result["errors"].append(f"{param_name} must be <= {param_config['max']}")
                        result["valid"] = False
                        
                elif param_type == "select":
                    options = param_config.get("options", [])
                    if value not in options:
                        result["errors"].append(f"{param_name} must be one of: {', '.join(options)}")
                        result["valid"] = False
                        
                elif param_type == "boolean":
                    if not isinstance(value, bool):
                        result["errors"].append(f"{param_name} must be true or false")
                        result["valid"] = False
        
        return result

    def generate_indicator_config_ui(self, indicator_name: str) -> Dict[str, Any]:
        """Generate UI configuration for an indicator's parameters"""
        indicator_params = self.get_indicator_parameters(indicator_name)
        if not indicator_params:
            return {"error": f"Unknown indicator: {indicator_name}"}
        
        ui_config = {
            "indicator_name": indicator_name,
            "category": indicator_params.get("category", "Unknown"),
            "description": indicator_params.get("description", ""),
            "parameters": []
        }
        
        parameters = indicator_params.get("parameters", {})
        for param_name, param_config in parameters.items():
            ui_param = {
                "name": param_name,
                "label": param_name.replace("_", " ").title(),
                "type": param_config.get("type"),
                "default": param_config.get("default"),
                "required": True
            }
            
            # Add type-specific properties
            if param_config.get("type") == "int":
                ui_param.update({
                    "min": param_config.get("min"),
                    "max": param_config.get("max"),
                    "step": param_config.get("step", 1)
                })
            elif param_config.get("type") == "float":
                ui_param.update({
                    "min": param_config.get("min"),
                    "max": param_config.get("max"),
                    "step": param_config.get("step", 0.1)
                })
            elif param_config.get("type") == "select":
                ui_param["options"] = param_config.get("options", [])
            elif param_config.get("type") == "multi_select":
                ui_param["options"] = param_config.get("options", [])
                ui_param["multiple"] = True
            
            ui_config["parameters"].append(ui_param)
        
        return ui_config

# Example usage
if __name__ == "__main__":
    # Test the EA generator
    generator = QNTIEAGenerator()
    
    # Create test profile
    profile = GenerationProfile(
        name="Test Profile",
        description="Test generation profile",
        strategy_type="trend_following",
        risk_level="moderate",
        account_size=10000,
        risk_per_trade=0.02,
        max_drawdown=0.15,
        symbol="EURUSD",
        timeframe="H1",
        generation_period="6_months",
        trading_direction="both",
        exit_strategy="hybrid",
        num_variants=5,
        optimization_level="basic"
    )
    
    # Generate EAs
    generator.generate_eas_from_profile(profile)
    
    # Monitor progress
    while generator.get_generation_status()["is_running"]:
        time.sleep(1)
        updates = generator.get_progress_updates()
        for update in updates:
            print(f"Progress: {update['message']}")
    
    print(f"Generation complete! Total EAs generated: {generator.get_generated_eas_count()}") 


# Compatibility classes for EA Integration System
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

class OptimizationMethod(Enum):
    """Optimization method types"""
    GRID_SEARCH = "grid_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class RobustnessTestType(Enum):
    """Robustness test types"""
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    OUT_OF_SAMPLE = "out_of_sample"

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class RobustnessConfig:
    """Robustness testing configuration"""
    tests: Optional[List[RobustnessTestType]] = None
    walk_forward_steps: int = 5
    monte_carlo_runs: int = 100
    sensitivity_range: float = 0.2
    out_of_sample_ratio: float = 0.2

@dataclass
class EATemplate:
    """EA Template for compatibility"""
    name: str
    description: str
    strategy_type: str
    entry_exit_rules: Optional[Dict[str, Any]] = None
    risk_management: Optional[Dict[str, Any]] = None
    indicators: Optional[List[str]] = None

@dataclass
class EAGenerationResult:
    """EA generation result"""
    ea_id: str
    success: bool
    ea_code: str = ""
    optimization_results: Optional[Dict[str, Any]] = None
    robustness_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None