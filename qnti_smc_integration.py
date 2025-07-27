#!/usr/bin/env python3
"""
QNTI Smart Money Concepts Integration Module

Integrates SMC analysis with the QNTI trading system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import json

from qnti_smart_money_concepts import SmartMoneyConcepts, SMCResult
from qnti_main_system import QNTIMainSystem

logger = logging.getLogger(__name__)

class QNTISMCIntegration:
    """
    Integration class for Smart Money Concepts in QNTI Trading System
    """
    
    def __init__(self, qnti_system: Optional[QNTIMainSystem] = None):
        self.qnti_system = qnti_system
        self.smc_analyzers = {}  # Symbol -> SMC analyzer mapping
        self.smc_results = {}    # Symbol -> SMC results mapping
        self.last_analysis_time = {}  # Symbol -> last analysis timestamp
        
        # SMC configuration per symbol type
        self.smc_configs = {
            'forex': {
                'swing_length': 50,
                'internal_length': 5,
                'equal_hl_threshold': 0.0001,  # 1 pip for forex
                'order_block_count': 5,
                'atr_period': 200
            },
            'gold': {
                'swing_length': 30,
                'internal_length': 5,
                'equal_hl_threshold': 0.5,  # $0.50 for gold
                'order_block_count': 5,
                'atr_period': 200
            },
            'crypto': {
                'swing_length': 40,
                'internal_length': 5,
                'equal_hl_threshold': 50,  # $50 for Bitcoin
                'order_block_count': 5,
                'atr_period': 200
            },
            'indices': {
                'swing_length': 50,
                'internal_length': 5,
                'equal_hl_threshold': 5,  # 5 points for indices
                'order_block_count': 5,
                'atr_period': 200
            }
        }
        
    def get_symbol_type(self, symbol: str) -> str:
        """Determine symbol type for appropriate SMC configuration"""
        symbol_upper = symbol.upper()
        
        if any(pair in symbol_upper for pair in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            if len(symbol_upper) == 6:  # Standard forex pair
                return 'forex'
        
        if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 'gold'
        
        if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'BITCOIN', 'ETHEREUM']):
            return 'crypto'
        
        if any(index in symbol_upper for index in ['US30', 'US500', 'US100', 'NAS', 'SPX', 'DJI']):
            return 'indices'
        
        return 'forex'  # Default
    
    def get_smc_analyzer(self, symbol: str) -> SmartMoneyConcepts:
        """Get or create SMC analyzer for symbol"""
        if symbol not in self.smc_analyzers:
            symbol_type = self.get_symbol_type(symbol)
            config = self.smc_configs[symbol_type]
            
            self.smc_analyzers[symbol] = SmartMoneyConcepts(**config)
            logger.info(f"Created SMC analyzer for {symbol} (type: {symbol_type})")
        
        return self.smc_analyzers[symbol]
    
    async def analyze_symbol(self, symbol: str, timeframe: str = 'H1', bars_count: int = 1000) -> Optional[SMCResult]:
        """
        Analyze SMC for a specific symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis (H1, H4, D1, etc.)
            bars_count: Number of bars to analyze
            
        Returns:
            SMCResult or None if analysis failed
        """
        try:
            if not self.qnti_system:
                logger.error("QNTI system not available for SMC analysis")
                return None
            
            # Get historical data from MT5
            data = await self._get_historical_data(symbol, timeframe, bars_count)
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for SMC analysis of {symbol}")
                return None
            
            # Get SMC analyzer and perform analysis
            analyzer = self.get_smc_analyzer(symbol)
            result = analyzer.analyze(data)
            
            # Store results
            self.smc_results[symbol] = result
            self.last_analysis_time[symbol] = datetime.now()
            
            logger.info(f"SMC analysis completed for {symbol} on {timeframe}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing SMC for {symbol}: {e}")
            return None
    
    async def _get_historical_data(self, symbol: str, timeframe: str, bars_count: int) -> Optional[pd.DataFrame]:
        """Get historical data from MT5 using direct MT5 integration"""
        try:
            import MetaTrader5 as mt5
            
            # Convert timeframe to MT5 constants
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # First, try to get data through QNTI system MT5 bridge
            if self.qnti_system and hasattr(self.qnti_system, 'mt5_bridge'):
                mt5_bridge = self.qnti_system.mt5_bridge
                if mt5_bridge and mt5_bridge.connection_status.value == "connected":
                    logger.info(f"Using QNTI MT5 bridge for {symbol} data")
                    
                    # Get data directly using MT5 copy_rates
                    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars_count)
                    
                    if rates is not None and len(rates) > 0:
                        # Convert MT5 rates to DataFrame
                        df = pd.DataFrame(rates)
                        
                        # Convert time to datetime
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        
                        # Rename tick_volume to volume if needed
                        if 'tick_volume' in df.columns:
                            df['volume'] = df['tick_volume']
                        
                        # Set timestamp as index
                        df.set_index('timestamp', inplace=True)
                        
                        # Ensure required columns exist
                        required_columns = ['open', 'high', 'low', 'close']
                        if all(col in df.columns for col in required_columns):
                            logger.info(f"Retrieved {len(df)} bars of live MT5 data for {symbol}")
                            return df[['open', 'high', 'low', 'close', 'volume'] if 'volume' in df.columns else ['open', 'high', 'low', 'close']]
            
            # Fallback: Try direct MT5 connection if QNTI bridge is not available
            logger.warning(f"QNTI MT5 bridge not available, trying direct MT5 connection for {symbol}")
            
            # Initialize MT5 if not already initialized
            if not mt5.initialize():
                logger.error("Failed to initialize MT5 for direct access")
                return None
            
            # Get data directly from MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars_count)
            
            if rates is not None and len(rates) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Ensure volume column
                if 'tick_volume' in df.columns:
                    df['volume'] = df['tick_volume']
                
                logger.info(f"Retrieved {len(df)} bars of direct MT5 data for {symbol}")
                return df[['open', 'high', 'low', 'close', 'volume'] if 'volume' in df.columns else ['open', 'high', 'low', 'close']]
            
            logger.warning(f"No data retrieved from MT5 for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting MT5 historical data for {symbol}: {e}")
            return None
    
    async def analyze_multiple_symbols(self, symbols: List[str], timeframe: str = 'H1') -> Dict[str, SMCResult]:
        """Analyze SMC for multiple symbols"""
        results = {}
        tasks = []
        
        for symbol in symbols:
            task = self.analyze_symbol(symbol, timeframe)
            tasks.append((symbol, task))
        
        # Run analyses concurrently
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Error in SMC analysis for {symbol}: {e}")
        
        return results
    
    def get_smc_summary(self, symbol: str) -> Optional[Dict]:
        """Get SMC summary for a symbol"""
        if symbol not in self.smc_results:
            return None
        
        analyzer = self.get_smc_analyzer(symbol)
        # Temporarily set the result to get summary
        analyzer.result = self.smc_results[symbol]
        return analyzer.get_summary()
    
    def get_trading_signals(self, symbol: str) -> Optional[Dict]:
        """Get trading signals based on SMC analysis"""
        if symbol not in self.smc_results:
            return None
        
        analyzer = self.get_smc_analyzer(symbol)
        analyzer.result = self.smc_results[symbol]
        signals = analyzer.get_trading_signals()
        
        # Add symbol-specific context
        signals['symbol'] = symbol
        signals['symbol_type'] = self.get_symbol_type(symbol)
        signals['last_analysis'] = self.last_analysis_time.get(symbol)
        
        return signals
    
    def get_all_smc_data(self) -> Dict[str, Any]:
        """Get all SMC data for dashboard display"""
        dashboard_data = {
            'symbols_analyzed': list(self.smc_results.keys()),
            'last_update': datetime.now().isoformat(),
            'summary': {},
            'signals': {},
            'alerts': {}
        }
        
        for symbol in self.smc_results.keys():
            # Get summary
            summary = self.get_smc_summary(symbol)
            if summary:
                dashboard_data['summary'][symbol] = summary
            
            # Get signals
            signals = self.get_trading_signals(symbol)
            if signals:
                dashboard_data['signals'][symbol] = signals
            
            # Extract alerts
            if summary and summary.get('alerts'):
                dashboard_data['alerts'][symbol] = [
                    alert_type for alert_type, is_active in summary['alerts'].items() 
                    if is_active
                ]
        
        return dashboard_data
    
    def get_key_levels_for_symbol(self, symbol: str) -> List[Dict]:
        """Get key levels (support/resistance) for a symbol"""
        signals = self.get_trading_signals(symbol)
        if not signals:
            return []
        
        levels = []
        
        # Add key levels from signals
        for level_data in signals.get('key_levels', []):
            levels.append({
                'level': level_data['level'],
                'type': level_data['type'],
                'importance': level_data['importance'],
                'source': 'swing_point'
            })
        
        # Add order block levels
        for ob in signals.get('order_blocks', []):
            levels.append({
                'level': (ob['high'] + ob['low']) / 2,  # Midpoint
                'type': 'support' if ob['bias'] == 'Bullish' else 'resistance',
                'importance': 'medium',
                'source': 'order_block',
                'range': [ob['low'], ob['high']]
            })
        
        # Sort by importance and level
        importance_order = {'high': 3, 'medium': 2, 'low': 1}
        levels.sort(key=lambda x: (importance_order.get(x['importance'], 0), x['level']), reverse=True)
        
        return levels
    
    def should_analyze(self, symbol: str, max_age_minutes: int = 30) -> bool:
        """Check if symbol needs fresh SMC analysis"""
        if symbol not in self.last_analysis_time:
            return True
        
        time_since_analysis = datetime.now() - self.last_analysis_time[symbol]
        return time_since_analysis.total_seconds() > (max_age_minutes * 60)
    
    def get_mock_smc_data(self, symbol: str) -> Dict:
        """Get mock SMC data for testing purposes"""
        import random
        
        # Generate realistic mock data based on symbol type
        base_prices = {
            'EURUSD': 1.0900, 'GBPUSD': 1.2500, 'USDJPY': 149.50, 'USDCHF': 0.8900, 
            'AUDUSD': 0.6600, 'USDCAD': 1.3600, 'NZDUSD': 0.6100,
            'EURJPY': 163.00, 'GBPJPY': 187.00, 'EURGBP': 0.8700, 'EURAUD': 1.6500,
            'GBPAUD': 1.9000, 'AUDCAD': 0.8980, 'AUDCHF': 0.5870, 'CADJPY': 110.00,
            'CHFJPY': 168.00, 'EURCHF': 0.9700, 'EURCAD': 1.4800, 'GBPCAD': 1.7000,
            'GBPCHF': 1.1100, 'NZDCAD': 0.8300, 'NZDJPY': 91.50, 'AUDNZD': 1.0800,
            'XAUUSD': 2030.00, 'XAGUSD': 24.50, 'GOLD': 2030.00, 'SILVER': 24.50,
            'BTCUSD': 42000.00, 'ETHUSD': 2500.00, 'LTCUSD': 70.00, 'XRPUSD': 0.60, 'ADAUSD': 0.45,
            'US30Cash': 34500.00, 'US500Cash': 4450.00, 'US100Cash': 15200.00, 
            'DE40Cash': 16800.00, 'UK100Cash': 7600.00, 'JP225Cash': 33000.00,
            'AUS200Cash': 7300.00, 'HK50Cash': 17500.00, 'USDX': 103.50,
            'WTICASH': 78.50, 'BRENTCASH': 82.00, 'NATGASCASH': 2.85
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate random but realistic SMC data
        random.seed(hash(symbol))  # Consistent data per symbol
        
        trends = ['BULLISH', 'BEARISH', 'NEUTRAL']
        swing_trend = random.choice(trends)
        internal_trend = random.choice(trends)
        
        # Calculate swing high/low based on base price and volatility
        volatility = 0.02 if 'USD' in symbol and symbol not in ['BTCUSD', 'ETHUSD'] else 0.05
        if symbol in ['XAUUSD', 'GOLD']:
            volatility = 0.01
        elif symbol in ['BTCUSD', 'ETHUSD']:
            volatility = 0.08
        elif 'Cash' in symbol:  # Indices
            volatility = 0.015
            
        swing_high = base_price * (1 + volatility * random.uniform(0.5, 1.5))
        swing_low = base_price * (1 - volatility * random.uniform(0.5, 1.5))
        
        mock_data = {
            'swing_trend': swing_trend,
            'internal_trend': internal_trend,
            'swing_high': round(swing_high, 5 if base_price < 10 else 2),
            'swing_low': round(swing_low, 5 if base_price < 10 else 2),
            'order_blocks_count': { 
                'swing': random.randint(2, 6), 
                'internal': random.randint(3, 8) 
            },
            'fair_value_gaps_count': random.randint(1, 12),
            'equal_highs_count': random.randint(0, 5),
            'equal_lows_count': random.randint(0, 5),
            'structure_breakouts_count': random.randint(1, 4),
            'zones': {
                'premium': [round(base_price * 1.002, 5), round(base_price * 1.005, 5)],
                'equilibrium': [round(base_price * 0.998, 5), round(base_price * 1.002, 5)],
                'discount': [round(base_price * 0.995, 5), round(base_price * 0.998, 5)]
            },
            'alerts': { 
                'swing_bullish_bos': swing_trend == 'BULLISH' and random.choice([True, False]),
                'equal_highs': random.choice([True, False]),
                'swing_ob': random.randint(0, 3),
                'internal_ob': random.randint(0, 5),
                'fvg': random.randint(0, 2)
            },
            'last_price': round(base_price, 5 if base_price < 10 else 2),
            'data_source': 'mock_data'
        }
        
        return { 'summary': mock_data }
    
    def get_symbol_type(self, symbol: str) -> str:
        """Get symbol type for classification"""
        if 'USD' in symbol and symbol != 'XAUUSD' and symbol != 'XAGUSD' and symbol != 'BTCUSD':
            return 'forex'
        elif symbol in ['XAUUSD', 'XAGUSD']:
            return 'metals'
        elif symbol == 'BTCUSD':
            return 'crypto'
        elif 'Cash' in symbol:
            return 'indices'
        else:
            return 'forex'
    
    async def refresh_analysis(self, symbols: List[str] = None, force: bool = False) -> Dict[str, bool]:
        """Refresh SMC analysis for symbols"""
        if symbols is None:
            symbols = list(self.smc_results.keys())
        
        refresh_results = {}
        
        for symbol in symbols:
            if force or self.should_analyze(symbol):
                try:
                    result = await self.analyze_symbol(symbol)
                    refresh_results[symbol] = result is not None
                except Exception as e:
                    logger.error(f"Error refreshing SMC analysis for {symbol}: {e}")
                    refresh_results[symbol] = False
            else:
                refresh_results[symbol] = True  # No refresh needed
        
        return refresh_results


# Integration with QNTI web routes
def create_smc_routes():
    """Create web routes for SMC functionality"""
    
    routes = []
    
    # This would be integrated into qnti_web.py
    smc_integration_code = '''
    
# Add to qnti_web.py:

from qnti_smc_integration import QNTISMCIntegration

# Initialize SMC integration
smc_integration = QNTISMCIntegration(qnti_system)

@app.route('/api/smc/analyze/<symbol>')
async def analyze_smc(symbol):
    """Analyze SMC for a specific symbol"""
    try:
        timeframe = request.args.get('timeframe', 'H1')
        result = await smc_integration.analyze_symbol(symbol, timeframe)
        
        if result:
            summary = smc_integration.get_smc_summary(symbol)
            signals = smc_integration.get_trading_signals(symbol)
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'summary': summary,
                'signals': signals
            })
        else:
            return jsonify({'success': False, 'error': 'Analysis failed'})
            
    except Exception as e:
        logger.error(f"Error in SMC analysis API: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/smc/dashboard')
async def smc_dashboard():
    """Get SMC dashboard data"""
    try:
        # Get main trading symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
        
        # Refresh analysis if needed
        await smc_integration.refresh_analysis(symbols)
        
        # Get dashboard data
        dashboard_data = smc_integration.get_all_smc_data()
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
        
    except Exception as e:
        logger.error(f"Error in SMC dashboard API: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/smc/levels/<symbol>')
async def get_smc_levels(symbol):
    """Get key levels for a symbol"""
    try:
        levels = smc_integration.get_key_levels_for_symbol(symbol)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'levels': levels
        })
        
    except Exception as e:
        logger.error(f"Error getting SMC levels: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/smc/refresh')
async def refresh_smc():
    """Refresh SMC analysis for all symbols"""
    try:
        symbols = request.json.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'])
        force = request.json.get('force', False)
        
        results = await smc_integration.refresh_analysis(symbols, force)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error refreshing SMC: {e}")
        return jsonify({'success': False, 'error': str(e)})

'''
    
    return routes, smc_integration_code


if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        smc_integration = QNTISMCIntegration()
        
        # Create sample data for testing
        import numpy as np
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        close_prices = 1.1000 + np.cumsum(np.random.randn(500) * 0.0001)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(500) * 0.00005,
            'high': close_prices + np.abs(np.random.randn(500) * 0.0001),
            'low': close_prices - np.abs(np.random.randn(500) * 0.0001),
            'close': close_prices,
            'volume': np.random.randint(100, 1000, 500)
        })
        
        # Test analyzer
        analyzer = smc_integration.get_smc_analyzer('EURUSD')
        result = analyzer.analyze(df)
        
        smc_integration.smc_results['EURUSD'] = result
        smc_integration.last_analysis_time['EURUSD'] = datetime.now()
        
        # Test outputs
        summary = smc_integration.get_smc_summary('EURUSD')
        signals = smc_integration.get_trading_signals('EURUSD')
        levels = smc_integration.get_key_levels_for_symbol('EURUSD')
        
        print("=== SMC Integration Test ===")
        print(f"Summary: {summary}")
        print(f"Signals: {signals}")
        print(f"Key Levels: {len(levels)}")
        
        return summary, signals, levels
    
    # Run test
    asyncio.run(test_integration()) 