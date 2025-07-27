#!/usr/bin/env python3
"""
QNTI Unified Automation API Integration
Centralized management of all automated trading strategies with MT5 integration
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Flask, request, jsonify

# Import all strategy implementations
from rsi_divergence_ea import RSIDivergenceEA
from macd_advanced_ea import MACDAdvancedEA
from bollinger_bands_ea import BollingerBandsEA
from ichimoku_cloud_ea import IchimokuCloudEA
from nr4_nr7_breakout_ea import NR4NR7BreakoutEA
from supertrend_dual_ea import SuperTrendDualEA
from qnti_smc_automation import QNTISMCAutomation
from qnti_persistence_manager import get_persistence_manager

# Import centralized logging
from qnti_logging_utils import get_qnti_logger
logger = get_qnti_logger(__name__)

class QNTIUnifiedAutomationManager:
    """
    Unified Automation Manager
    
    Manages all automated trading strategies:
    - RSI with Divergence Detection
    - MACD Advanced
    - Bollinger Bands
    - Ichimoku Cloud
    - NR4/NR7 Breakout
    - SuperTrend Dual
    - SMC Automation (Smart Money Concepts)
    """
    
    def __init__(self, qnti_main_system):
        self.qnti_main_system = qnti_main_system
        # Handle case where qnti_main_system might not be fully initialized
        self.trade_manager = getattr(qnti_main_system, 'trade_manager', None)
        self.mt5_bridge = getattr(qnti_main_system, 'mt5_bridge', None)
        
        # Initialize persistence manager
        self.persistence = get_persistence_manager()
        
        # Initialize all strategy instances
        self.strategies = {
            "rsi_divergence": RSIDivergenceEA(),
            "macd_advanced": MACDAdvancedEA(),
            "bollinger_bands": BollingerBandsEA(),
            "ichimoku_cloud": IchimokuCloudEA(),
            "nr4_nr7_breakout": NR4NR7BreakoutEA(),
            "supertrend_dual": SuperTrendDualEA(),
            "smc_automation": QNTISMCAutomation(qnti_main_system)
        }
        
        # Set QNTI integration for all strategies (except SMC which handles its own integration)
        for name, strategy in self.strategies.items():
            if name != "smc_automation" and hasattr(strategy, 'set_qnti_integration') and self.trade_manager and self.mt5_bridge:
                try:
                    strategy.set_qnti_integration(self.trade_manager, self.mt5_bridge, qnti_main_system)
                except Exception as e:
                    logger.warning(f"Could not set QNTI integration for {name}: {e}")
        
        # Strategy status tracking
        self.strategy_status = {name: "stopped" for name in self.strategies.keys()}
        self.strategy_configs = {}
        
        # Global automation settings
        self.global_settings = {
            "auto_trading_enabled": False,
            "max_concurrent_strategies": 3,
            "risk_per_trade": 0.01,
            "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
            "timeframes": ["M15", "H1", "H4"]
        }
        
        # Restore previous states from persistence
        self.restore_strategy_states()
        
        logger.info("QNTI Unified Automation Manager initialized")
    
    def restore_strategy_states(self):
        """Restore strategy states from persistence"""
        try:
            logger.info("Restoring strategy states from persistence...")
            
            # Load all saved strategy states
            saved_states = self.persistence.load_all_strategy_states()
            
            for strategy_name, saved_state in saved_states.items():
                if strategy_name in self.strategies:
                    try:
                        # Restore strategy status
                        if saved_state.get('active', False):
                            self.strategy_status[strategy_name] = "running"
                        else:
                            self.strategy_status[strategy_name] = "stopped"
                        
                        # Restore strategy parameters
                        if saved_state.get('parameters'):
                            self.strategy_configs[strategy_name] = saved_state['parameters']
                            
                            # Apply restored parameters to strategy
                            if strategy_name == "smc_automation":
                                # SMC automation has different method names
                                self.strategies[strategy_name].update_settings(saved_state['parameters'])
                            elif hasattr(self.strategies[strategy_name], 'update_parameters'):
                                self.strategies[strategy_name].update_parameters(saved_state['parameters'])
                        
                        logger.info(f"Strategy state restored: {strategy_name}")
                        
                    except Exception as e:
                        logger.error(f"Error restoring strategy state {strategy_name}: {e}")
                else:
                    logger.warning(f"Saved strategy not found in current strategies: {strategy_name}")
            
            logger.info(f"Strategy state restoration completed. Restored {len(saved_states)} strategies.")
            
        except Exception as e:
            logger.error(f"Error restoring strategy states: {e}")

    def save_strategy_state(self, strategy_name: str):
        """Save strategy state to persistence"""
        try:
            if strategy_name in self.strategies:
                state = {
                    'active': self.strategy_status.get(strategy_name) == "running",
                    'parameters': self.strategy_configs.get(strategy_name, {}),
                    'last_signal_time': None,  # Could be enhanced to track last signal
                    'performance_metrics': {}  # Could be enhanced to track performance
                }
                
                success = self.persistence.save_strategy_state(strategy_name, state)
                if success:
                    logger.info(f"Strategy state saved: {strategy_name}")
                return success
            else:
                logger.warning(f"Strategy not found for saving: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving strategy state {strategy_name}: {e}")
            return False
    
    def start_strategy(self, strategy_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a specific trading strategy"""
        try:
            if strategy_name not in self.strategies:
                return {"success": False, "error": f"Unknown strategy: {strategy_name}"}
            
            if self.strategy_status[strategy_name] == "running":
                return {"success": False, "error": f"Strategy {strategy_name} is already running"}
            
            strategy = self.strategies[strategy_name]
            
            # Handle SMC automation differently (it has different method names)
            if strategy_name == "smc_automation":
                # Update SMC settings
                if config.get("parameters"):
                    strategy.update_settings(config["parameters"])
                
                # Start SMC automation
                strategy.start_automation()
            else:
                # Update strategy parameters
                if config.get("parameters"):
                    strategy.update_parameters(config["parameters"])
                
                # Get symbols to monitor
                symbols = config.get("symbols", self.global_settings["symbols"])
                
                # Start strategy monitoring
                strategy.start_monitoring(symbols)
            
            # Update status
            self.strategy_status[strategy_name] = "running"
            self.strategy_configs[strategy_name] = config
            
            # Save state to persistence
            self.save_strategy_state(strategy_name)
            
            logger.info(f"Started strategy: {strategy_name}")
            
            return {
                "success": True,
                "strategy": strategy_name,
                "status": "running",
                "symbols": config.get("symbols", self.global_settings["symbols"]),
                "config": config
            }
            
        except Exception as e:
            logger.error(f"Error starting strategy {strategy_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Stop a specific trading strategy"""
        try:
            if strategy_name not in self.strategies:
                return {"success": False, "error": f"Unknown strategy: {strategy_name}"}
            
            if self.strategy_status[strategy_name] == "stopped":
                return {"success": False, "error": f"Strategy {strategy_name} is already stopped"}
            
            strategy = self.strategies[strategy_name]
            
            # Handle SMC automation differently
            if strategy_name == "smc_automation":
                strategy.stop_automation()
            else:
                # Stop strategy monitoring
                strategy.stop_monitoring()
            
            # Update status
            self.strategy_status[strategy_name] = "stopped"
            
            # Save state to persistence
            self.save_strategy_state(strategy_name)
            
            logger.info(f"Stopped strategy: {strategy_name}")
            
            return {
                "success": True,
                "strategy": strategy_name,
                "status": "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error stopping strategy {strategy_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_strategy_status(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all strategies"""
        try:
            if strategy_name:
                if strategy_name not in self.strategies:
                    return {"success": False, "error": f"Unknown strategy: {strategy_name}"}
                
                strategy = self.strategies[strategy_name]
                
                # Handle SMC automation differently
                if strategy_name == "smc_automation":
                    detailed_status = strategy.get_automation_status()
                else:
                    detailed_status = strategy.get_status()
                
                return {
                    "success": True,
                    "strategy": strategy_name,
                    "status": self.strategy_status[strategy_name],
                    "details": detailed_status
                }
            else:
                # Return status of all strategies
                all_status = {}
                for name, strategy in self.strategies.items():
                    # Handle SMC automation differently
                    if name == "smc_automation":
                        detailed_status = strategy.get_automation_status()
                    else:
                        detailed_status = strategy.get_status()
                    
                    all_status[name] = {
                        "status": self.strategy_status[name],
                        "details": detailed_status
                    }
                
                return {
                    "success": True,
                    "strategies": all_status,
                    "global_settings": self.global_settings
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {"success": False, "error": str(e)}
    
    def update_global_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update global automation settings"""
        try:
            self.global_settings.update(settings)
            logger.info(f"Updated global settings: {settings}")
            
            return {
                "success": True,
                "settings": self.global_settings
            }
            
        except Exception as e:
            logger.error(f"Error updating global settings: {e}")
            return {"success": False, "error": str(e)}
    
    def get_automation_metrics(self) -> Dict[str, Any]:
        """Get overall automation performance metrics"""
        try:
            running_strategies = sum(1 for status in self.strategy_status.values() if status == "running")
            total_signals = 0
            total_trades = 0
            
            # Collect metrics from all strategies
            strategy_metrics = {}
            for name, strategy in self.strategies.items():
                status = strategy.get_status()
                signals_count = status.get("signals_count", 0)
                total_signals += signals_count
                
                strategy_metrics[name] = {
                    "running": self.strategy_status[name] == "running",
                    "signals_generated": signals_count,
                    "current_position": status.get("position", 0),
                    "last_signal_time": status.get("last_signal", {}).get("timestamp") if status.get("last_signal") else None
                }
            
            # Get trade manager metrics
            if self.trade_manager:
                trade_health = self.trade_manager.get_system_health()
                total_trades = trade_health.get("open_trades", 0)
            
            return {
                "success": True,
                "metrics": {
                    "active_strategies": running_strategies,
                    "total_strategies": len(self.strategies),
                    "total_signals_generated": total_signals,
                    "active_trades": total_trades,
                    "strategy_breakdown": strategy_metrics,
                    "last_updated": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting automation metrics: {e}")
            return {"success": False, "error": str(e)}

# Global manager instance
unified_manager = None

def integrate_unified_automation_with_qnti_web(app: Flask, qnti_main_system) -> bool:
    """Integrate unified automation API with QNTI web interface"""
    global unified_manager
    
    try:
        # Initialize the unified manager
        unified_manager = QNTIUnifiedAutomationManager(qnti_main_system)
        
        @app.route('/api/unified-automation/start', methods=['POST'])
        def start_automation():
            """Start automated trading strategy"""
            try:
                data = request.get_json() or {}
                strategy_name = data.get('strategy')
                config = data.get('config', {})
                
                if not strategy_name:
                    return jsonify({"success": False, "error": "Strategy name required"}), 400
                
                result = unified_manager.start_strategy(strategy_name, config)
                status_code = 200 if result["success"] else 400
                
                return jsonify(result), status_code
                
            except Exception as e:
                logger.error(f"Error in start_automation endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/api/unified-automation/stop', methods=['POST'])
        def stop_automation():
            """Stop automated trading strategy"""
            try:
                data = request.get_json() or {}
                strategy_name = data.get('strategy')
                
                if not strategy_name:
                    return jsonify({"success": False, "error": "Strategy name required"}), 400
                
                result = unified_manager.stop_strategy(strategy_name)
                status_code = 200 if result["success"] else 400
                
                return jsonify(result), status_code
                
            except Exception as e:
                logger.error(f"Error in stop_automation endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/api/unified-automation/status', methods=['GET'])
        def get_automation_status():
            """Get automation status"""
            try:
                strategy_name = request.args.get('strategy')
                result = unified_manager.get_strategy_status(strategy_name)
                
                status_code = 200 if result["success"] else 400
                return jsonify(result), status_code
                
            except Exception as e:
                logger.error(f"Error in get_automation_status endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/api/unified-automation/metrics', methods=['GET'])
        def get_automation_metrics():
            """Get automation performance metrics"""
            try:
                result = unified_manager.get_automation_metrics()
                status_code = 200 if result["success"] else 400
                
                return jsonify(result), status_code
                
            except Exception as e:
                logger.error(f"Error in get_automation_metrics endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/api/unified-automation/settings', methods=['POST'])
        def update_global_settings():
            """Update global automation settings"""
            try:
                data = request.get_json() or {}
                result = unified_manager.update_global_settings(data)
                
                status_code = 200 if result["success"] else 400
                return jsonify(result), status_code
                
            except Exception as e:
                logger.error(f"Error in update_global_settings endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        @app.route('/api/unified-automation/strategies', methods=['GET'])
        def list_available_strategies():
            """List all available strategies"""
            try:
                strategies_info = {
                    "rsi_divergence": {
                        "name": "RSI with Divergence Detection",
                        "description": "RSI oscillator with bullish/bearish divergence detection",
                        "parameters": {
                            "rsi_period": {"type": "number", "default": 14, "min": 2, "max": 50},
                            "rsi_overbought": {"type": "number", "default": 70, "min": 50, "max": 90},
                            "rsi_oversold": {"type": "number", "default": 30, "min": 10, "max": 50},
                            "divergence_lookback": {"type": "number", "default": 20, "min": 10, "max": 50},
                            "min_divergence_strength": {"type": "number", "default": 0.6, "min": 0.1, "max": 1.0}
                        }
                    },
                    "macd_advanced": {
                        "name": "MACD Advanced",
                        "description": "MACD with histogram analysis and zero line crossovers",
                        "parameters": {
                            "fast_period": {"type": "number", "default": 12, "min": 5, "max": 50},
                            "slow_period": {"type": "number", "default": 26, "min": 10, "max": 100},
                            "signal_period": {"type": "number", "default": 9, "min": 3, "max": 30},
                            "zero_cross_enabled": {"type": "boolean", "default": True},
                            "histogram_threshold": {"type": "number", "default": 0.0001, "min": 0.00001, "max": 0.001}
                        }
                    },
                    "bollinger_bands": {
                        "name": "Bollinger Bands",
                        "description": "Mean reversion and breakout trading with Bollinger Bands",
                        "parameters": {
                            "period": {"type": "number", "default": 20, "min": 10, "max": 50},
                            "std_dev": {"type": "number", "default": 2.0, "min": 1.0, "max": 3.0},
                            "squeeze_threshold": {"type": "number", "default": 0.1, "min": 0.05, "max": 0.3},
                            "breakout_confirmation": {"type": "boolean", "default": True},
                            "mean_reversion_mode": {"type": "boolean", "default": True}
                        }
                    },
                    "ichimoku_cloud": {
                        "name": "Ichimoku Cloud",
                        "description": "Complete Ichimoku system with cloud analysis",
                        "parameters": {
                            "tenkan_period": {"type": "number", "default": 9, "min": 5, "max": 20},
                            "kijun_period": {"type": "number", "default": 26, "min": 15, "max": 50},
                            "senkou_span_b_period": {"type": "number", "default": 52, "min": 30, "max": 100},
                            "chikou_confirmation": {"type": "boolean", "default": True},
                            "cloud_filter": {"type": "boolean", "default": True}
                        }
                    },
                    "nr4_nr7_breakout": {
                        "name": "NR4/NR7 Breakout",
                        "description": "Narrow Range breakout trading with volume confirmation",
                        "parameters": {
                            "enable_nr4": {"type": "boolean", "default": True},
                            "enable_nr7": {"type": "boolean", "default": True},
                            "enable_nr14": {"type": "boolean", "default": False},
                            "volume_confirmation": {"type": "boolean", "default": True},
                            "breakout_buffer": {"type": "number", "default": 0.0001, "min": 0.00001, "max": 0.001},
                            "max_wait_hours": {"type": "number", "default": 24, "min": 1, "max": 72}
                        }
                    },
                    "supertrend_dual": {
                        "name": "SuperTrend Dual",
                        "description": "Dual SuperTrend system with standard and centerline indicators",
                        "parameters": {
                            "st_period": {"type": "number", "default": 7, "min": 3, "max": 20},
                            "st_multiplier": {"type": "number", "default": 7.0, "min": 1.0, "max": 15.0},
                            "cl_period": {"type": "number", "default": 22, "min": 10, "max": 50},
                            "cl_multiplier": {"type": "number", "default": 3.0, "min": 1.0, "max": 10.0},
                            "cl_use_wicks": {"type": "boolean", "default": False}
                        }
                    },
                    "smc_automation": {
                        "name": "SMC Automation ⚡ LIVE",
                        "description": "Smart Money Concepts automation with order blocks, FVGs, and structure analysis - FULLY INTEGRATED WITH MT5",
                        "parameters": {
                            "auto_trading_enabled": {"type": "boolean", "default": False},
                            "max_risk_per_trade": {"type": "number", "default": 2.0, "min": 0.5, "max": 5.0},
                            "min_confidence": {"type": "number", "default": 0.7, "min": 0.1, "max": 1.0},
                            "premium_threshold": {"type": "number", "default": 0.7, "min": 0.5, "max": 0.9},
                            "discount_threshold": {"type": "number", "default": 0.3, "min": 0.1, "max": 0.5},
                            "alert_notifications": {"type": "boolean", "default": True}
                        }
                    }
                }
                
                return jsonify({
                    "success": True,
                    "strategies": strategies_info,
                    "total_count": len(strategies_info)
                }), 200
                
            except Exception as e:
                logger.error(f"Error in list_available_strategies endpoint: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        # Persistence Management API Endpoints
        @app.route('/api/unified-automation/persistence/status', methods=['GET'])
        def get_persistence_status():
            """Get persistence system status"""
            try:
                status = unified_manager.persistence.get_system_status()
                return jsonify({"success": True, "status": status}), 200
            except Exception as e:
                logger.error(f"Error getting persistence status: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/unified-automation/persistence/backup', methods=['POST'])
        def create_backup():
            """Create a system backup"""
            try:
                data = request.get_json() or {}
                backup_name = data.get('backup_name')
                
                backup_path = unified_manager.persistence.create_backup(backup_name)
                if backup_path:
                    return jsonify({
                        "success": True,
                        "backup_path": backup_path,
                        "message": "Backup created successfully"
                    }), 200
                else:
                    return jsonify({"success": False, "error": "Failed to create backup"}), 500
                    
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/unified-automation/persistence/restore', methods=['POST'])
        def restore_backup():
            """Restore from a backup"""
            try:
                data = request.get_json()
                if not data or 'backup_name' not in data:
                    return jsonify({"success": False, "error": "backup_name is required"}), 400
                
                success = unified_manager.persistence.restore_from_backup(data['backup_name'])
                if success:
                    return jsonify({
                        "success": True,
                        "message": "System restored from backup successfully"
                    }), 200
                else:
                    return jsonify({"success": False, "error": "Failed to restore from backup"}), 500
                    
            except Exception as e:
                logger.error(f"Error restoring backup: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/unified-automation/persistence/preferences', methods=['GET', 'POST'])
        def dashboard_preferences():
            """Get or set dashboard preferences"""
            try:
                if request.method == 'GET':
                    preferences = unified_manager.persistence.load_dashboard_preferences()
                    return jsonify({"success": True, "preferences": preferences}), 200
                    
                elif request.method == 'POST':
                    data = request.get_json()
                    if not data:
                        return jsonify({"success": False, "error": "No data provided"}), 400
                    
                    success = unified_manager.persistence.save_dashboard_preferences(data)
                    if success:
                        return jsonify({
                            "success": True,
                            "message": "Dashboard preferences saved"
                        }), 200
                    else:
                        return jsonify({"success": False, "error": "Failed to save preferences"}), 500
                        
            except Exception as e:
                logger.error(f"Error handling dashboard preferences: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/unified-automation/persistence/user-settings', methods=['GET', 'POST'])
        def user_settings():
            """Get or set user settings"""
            try:
                if request.method == 'GET':
                    settings = unified_manager.persistence.load_user_settings()
                    return jsonify({"success": True, "settings": settings}), 200
                    
                elif request.method == 'POST':
                    data = request.get_json()
                    if not data:
                        return jsonify({"success": False, "error": "No data provided"}), 400
                    
                    success = unified_manager.persistence.save_user_settings(data)
                    if success:
                        return jsonify({
                            "success": True,
                            "message": "User settings saved"
                        }), 200
                    else:
                        return jsonify({"success": False, "error": "Failed to save settings"}), 500
                        
            except Exception as e:
                logger.error(f"Error handling user settings: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        logger.info("Unified automation API integrated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating unified automation API: {e}")
        return False 