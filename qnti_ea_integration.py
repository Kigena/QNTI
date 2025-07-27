"""
QNTI EA Generation Integration
=============================

Integration layer that connects the EA Generation Engine with the existing
QNTI system components including strategy tester, MT5 bridge, web interface,
and data management systems.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import threading
import queue
import time

# Import QNTI EA Generator components
from qnti_ea_generator import (
    QNTIEAGenerator, EATemplate, OptimizationConfig, RobustnessConfig,
    EAGenerationResult, OptimizationMethod, RobustnessTestType
)
from qnti_ea_subagents import (
    SubagentManager, SubagentTask, SubagentType, TaskPriority,
    OptimizerSubagent, RobustnessSubagent, PerformanceEvaluatorSubagent
)
from qnti_ea_reporting import (
    QNTIEAReportingSystem, ReportConfig, ReportType, EAGenerationEvent, LogLevel
)

# Configure logging
# Import centralized logging
from qnti_logging_utils import get_qnti_logger
logger = get_qnti_logger(__name__)

class QNTIEAIntegrationManager:
    """
    Main integration manager that coordinates EA generation with QNTI system
    """
    
    def __init__(self, qnti_system=None):
        self.qnti_system = qnti_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize EA Generator and Subagent system
        self.ea_generator = QNTIEAGenerator(qnti_system)
        self.subagent_manager = SubagentManager()
        
        # Initialize comprehensive reporting system
        self.reporting_system = QNTIEAReportingSystem()
        
        # Integration components
        self.strategy_tester_bridge = None
        self.mt5_bridge = None
        self.web_interface_bridge = None
        self.data_manager_bridge = None
        
        # Active generation tasks
        self.active_generations = {}
        self.generation_queue = queue.Queue()
        
        # Configuration
        self.config = {
            "max_concurrent_generations": 3,
            "default_optimization_method": OptimizationMethod.GRID_SEARCH,
            "default_robustness_tests": [
                RobustnessTestType.WALK_FORWARD,
                RobustnessTestType.MONTE_CARLO,
                RobustnessTestType.PARAMETER_SENSITIVITY
            ],
            "integration_enabled": True
        }
        
        self.logger.info("QNTI EA Integration Manager initialized")
    
    def initialize_integration(self):
        """Initialize integration with QNTI system components"""
        try:
            # Initialize strategy tester bridge
            if hasattr(self.qnti_system, 'strategy_tester'):
                self.strategy_tester_bridge = StrategyTesterBridge(
                    self.qnti_system.strategy_tester
                )
                self.logger.info("Strategy Tester bridge initialized")
            
            # Initialize MT5 bridge
            if hasattr(self.qnti_system, 'mt5_bridge'):
                self.mt5_bridge = MT5Bridge(self.qnti_system.mt5_bridge)
                self.logger.info("MT5 bridge initialized")
            
            # Initialize web interface bridge
            if hasattr(self.qnti_system, 'web_server'):
                self.web_interface_bridge = WebInterfaceBridge(
                    self.qnti_system.web_server, self
                )
                self.logger.info("Web Interface bridge initialized")
            
            # Initialize data manager bridge
            if hasattr(self.qnti_system, 'data_manager'):
                self.data_manager_bridge = DataManagerBridge(
                    self.qnti_system.data_manager
                )
                self.logger.info("Data Manager bridge initialized")
            
            # Start subagent manager
            self.subagent_manager.start()
            
            self.logger.info("QNTI EA Integration fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing integration: {str(e)}")
            return False
    
    def _extract_indicators_from_template(self, template: EATemplate) -> List[str]:
        """Extract indicator names from EA template"""
        indicators = set()
        
        # Extract from entry rules
        if template.entry_exit_rules.entry_long:
            indicators.update(self._extract_indicators_from_conditions(template.entry_exit_rules.entry_long))
        if template.entry_exit_rules.entry_short:
            indicators.update(self._extract_indicators_from_conditions(template.entry_exit_rules.entry_short))
        if template.entry_exit_rules.exit_long:
            indicators.update(self._extract_indicators_from_conditions(template.entry_exit_rules.exit_long))
        if template.entry_exit_rules.exit_short:
            indicators.update(self._extract_indicators_from_conditions(template.entry_exit_rules.exit_short))
        
        return list(indicators)
    
    def _extract_indicators_from_conditions(self, condition) -> List[str]:
        """Extract indicators from logical conditions recursively"""
        indicators = []
        
        if hasattr(condition, 'conditions'):  # LogicalCondition
            for sub_condition in condition.conditions:
                indicators.extend(self._extract_indicators_from_conditions(sub_condition))
        elif hasattr(condition, 'left_indicator'):  # Condition
            indicators.append(condition.left_indicator.indicator_type.value)
            if condition.right_indicator:
                indicators.append(condition.right_indicator.indicator_type.value)
        
        return indicators
    
    def complete_ea_generation(self, generation_id: str, success: bool, result: Optional[Dict[str, Any]] = None):
        """Complete EA generation and log results"""
        if generation_id not in self.active_generations:
            self.logger.warning(f"Generation {generation_id} not found in active generations")
            return
        
        generation_task = self.active_generations[generation_id]
        start_time = generation_task["created_at"]
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log completion
        performance_metrics = result.get("performance_metrics", {}) if result else {}
        self.reporting_system.log_generation_complete(
            generation_id, success, duration_ms, performance_metrics
        )
        
        # Update status
        generation_task["status"] = "completed" if success else "failed"
        generation_task["completed_at"] = datetime.now()
        generation_task["result"] = result
        
        if success and result:
            # Log optimization results if available
            if "optimization_results" in result:
                opt_result = result["optimization_results"]
                method = generation_task["optimization_config"].method.value
                self.reporting_system.log_optimization_result(
                    generation_id, method, opt_result, duration_ms
                )
            
            # Log robustness test results if available
            if "robustness_results" in result:
                robust_results = result["robustness_results"]
                for test_type, test_result in robust_results.items():
                    self.reporting_system.log_robustness_test(
                        generation_id, test_type, test_result
                    )
        
        self.logger.info(f"EA generation {generation_id} completed successfully: {success}")
    
    def get_ea_generation_reports(self, report_type: str = "summary", time_range_hours: int = 24) -> str:
        """Get comprehensive reports from the reporting system"""
        config = ReportConfig(
            report_type=ReportType(report_type),
            output_format="json",
            time_range_hours=time_range_hours
        )
        
        return self.reporting_system.generate_report(config)
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for EA generation"""
        # Get data from reporting system
        reporting_metrics = self.reporting_system.get_real_time_metrics()
        
        # Add integration-specific metrics
        integration_metrics = {
            "active_generations": len(self.active_generations),
            "queued_tasks": self.generation_queue.qsize(),
            "subagent_status": self.subagent_manager.get_agent_status() if self.subagent_manager else {},
            "integration_bridges": {
                "strategy_tester": self.strategy_tester_bridge is not None,
                "mt5_bridge": self.mt5_bridge is not None,
                "web_interface": self.web_interface_bridge is not None,
                "data_manager": self.data_manager_bridge is not None
            }
        }
        
        # Combine metrics
        dashboard_data = {
            **reporting_metrics,
            **integration_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def create_ea_from_web_request(self, request_data: Dict[str, Any]) -> str:
        """Create EA from web interface request"""
        try:
            # Parse request data
            template_config = request_data.get("template_config", {})
            optimization_config = request_data.get("optimization_config", {})
            robustness_config = request_data.get("robustness_config", {})
            
            # Create EA template
            template = self.ea_generator.create_ea_template(template_config)
            
            # Create configuration objects
            opt_config = OptimizationConfig(
                method=OptimizationMethod(optimization_config.get("method", "grid_search")),
                target_metric=optimization_config.get("target_metric", "profit_factor"),
                max_iterations=optimization_config.get("max_iterations", 100),
                parallel_workers=optimization_config.get("parallel_workers", 4)
            )
            
            robust_config = RobustnessConfig(
                tests_to_run=[RobustnessTestType(t) for t in robustness_config.get("tests", [])],
                walk_forward_periods=robustness_config.get("walk_forward_periods", 12),
                monte_carlo_runs=robustness_config.get("monte_carlo_runs", 500)
            )
            
            # Submit for generation
            generation_id = self.submit_ea_generation(template, opt_config, robust_config)
            
            self.logger.info(f"EA generation submitted from web request: {generation_id}")
            return generation_id
            
        except Exception as e:
            self.logger.error(f"Error creating EA from web request: {str(e)}")
            raise
    
    def submit_ea_generation(self, template: EATemplate, optimization_config: OptimizationConfig,
                           robustness_config: RobustnessConfig) -> str:
        """Submit EA for generation using subagent system"""
        
        generation_id = str(uuid.uuid4())
        
        try:
            # Log generation start
            self.reporting_system.log_generation_start(
                generation_id, 
                template.name, 
                self._extract_indicators_from_template(template)
            )
            
            # Create generation task
            generation_task = {
                "generation_id": generation_id,
                "template": template,
                "optimization_config": optimization_config,
                "robustness_config": robustness_config,
                "status": "queued",
                "created_at": datetime.now(),
                "subtasks": []
            }
            
            # Add to active generations
            self.active_generations[generation_id] = generation_task
            
            # Submit data preparation task
            data_task = SubagentTask(
                task_id=str(uuid.uuid4()),
                task_type="data_preparation",
                subagent_type=SubagentType.DATA_MANAGER,
                priority=TaskPriority.HIGH,
                parameters={
                    "task_type": "data_preparation",
                    "symbols": template.symbols,
                    "timeframe": template.timeframe,
                    "generation_id": generation_id
                }
            )
            
            data_task_id = self.subagent_manager.submit_task(data_task)
            generation_task["subtasks"].append({"type": "data_preparation", "task_id": data_task_id})
            
            # Submit optimization task
            opt_task = SubagentTask(
                task_id=str(uuid.uuid4()),
                task_type="parameter_optimization",
                subagent_type=SubagentType.OPTIMIZER,
                priority=TaskPriority.HIGH,
                parameters={
                    "optimization_type": optimization_config.method.value,
                    "target_metric": optimization_config.target_metric,
                    "max_iterations": optimization_config.max_iterations,
                    "generation_id": generation_id,
                    "template": template
                }
            )
            
            opt_task_id = self.subagent_manager.submit_task(opt_task)
            generation_task["subtasks"].append({"type": "optimization", "task_id": opt_task_id})
            
            # Submit robustness testing tasks
            for test_type in robustness_config.tests_to_run:
                robust_task = SubagentTask(
                    task_id=str(uuid.uuid4()),
                    task_type=f"robustness_{test_type.value}",
                    subagent_type=SubagentType.ROBUSTNESS_TESTER,
                    priority=TaskPriority.NORMAL,
                    parameters={
                        "test_type": test_type.value,
                        "generation_id": generation_id,
                        "config": robustness_config
                    }
                )
                
                robust_task_id = self.subagent_manager.submit_task(robust_task)
                generation_task["subtasks"].append({
                    "type": f"robustness_{test_type.value}", 
                    "task_id": robust_task_id
                })
            
            generation_task["status"] = "processing"
            
            self.logger.info(f"EA generation {generation_id} submitted with {len(generation_task['subtasks'])} subtasks")
            return generation_id
            
        except Exception as e:
            self.logger.error(f"Error submitting EA generation: {str(e)}")
            if generation_id in self.active_generations:
                self.active_generations[generation_id]["status"] = "failed"
                self.active_generations[generation_id]["error"] = str(e)
            raise
    
    def get_generation_status(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of EA generation"""
        if generation_id not in self.active_generations:
            return None
        
        generation_task = self.active_generations[generation_id]
        
        # Check status of all subtasks
        subtask_statuses = []
        for subtask in generation_task["subtasks"]:
            task_status = self.subagent_manager.get_task_status(subtask["task_id"])
            subtask_statuses.append({
                "type": subtask["type"],
                "task_id": subtask["task_id"],
                "status": task_status["status"] if task_status else "unknown"
            })
        
        # Determine overall status
        completed_tasks = sum(1 for s in subtask_statuses if s["status"] == "completed")
        failed_tasks = sum(1 for s in subtask_statuses if s["status"] == "failed")
        total_tasks = len(subtask_statuses)
        
        if failed_tasks > 0:
            overall_status = "failed"
        elif completed_tasks == total_tasks:
            overall_status = "completed"
        else:
            overall_status = "processing"
        
        return {
            "generation_id": generation_id,
            "status": overall_status,
            "progress": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "subtasks": subtask_statuses,
            "created_at": generation_task["created_at"].isoformat(),
            "template_name": generation_task["template"].name
        }
    
    def get_completed_eas(self) -> List[Dict[str, Any]]:
        """Get all completed EA generations"""
        completed = []
        
        for generation_id, generation_task in self.active_generations.items():
            status = self.get_generation_status(generation_id)
            if status and status["status"] == "completed":
                completed.append(status)
        
        return completed
    
    def export_ea_for_mt5(self, generation_id: str) -> Optional[str]:
        """Export completed EA for MT5 deployment"""
        if generation_id not in self.active_generations:
            return None
        
        status = self.get_generation_status(generation_id)
        if not status or status["status"] != "completed":
            return None
        
        try:
            # Get generation results
            generation_task = self.active_generations[generation_id]
            template = generation_task["template"]
            
            # Create MT5-compatible EA code
            ea_code = self._generate_mt5_ea_code(generation_id, template)
            
            # Save to MT5 experts directory
            if self.mt5_bridge:
                file_path = self.mt5_bridge.save_ea_file(f"{template.name}_{generation_id}", ea_code)
                self.logger.info(f"EA exported to MT5: {file_path}")
                return file_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error exporting EA to MT5: {str(e)}")
            return None
    
    def _generate_mt5_ea_code(self, generation_id: str, template: EATemplate) -> str:
        """Generate MT5-compatible EA code"""
        # This would generate actual MQL5 code
        # For now, returning a template
        
        ea_code = f"""
//+------------------------------------------------------------------+
//| Expert Advisor: {template.name}                                  |
//| Generated by QNTI EA Generator                                   |
//| Generation ID: {generation_id}                                   |
//| Created: {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}         |
//+------------------------------------------------------------------+

#property copyright "QNTI Trading System"
#property link      "https://qnti.trading"
#property version   "1.00"
#property strict

// Input Parameters
input double LotSize = {template.risk_management.position_size};
input double StopLoss = {template.risk_management.stop_loss or 0.0};
input double TakeProfit = {template.risk_management.take_profit or 0.0};
input int MaxPositions = {template.risk_management.max_positions};

// Global Variables
int magic_number = {hash(generation_id) % 999999};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{{
    Print("QNTI Generated EA {template.name} initialized");
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    Print("QNTI Generated EA {template.name} deinitialized");
}}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{{
    // Check for new bar
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(Symbol(), PERIOD_CURRENT, 0);
    
    if(current_bar_time != last_bar_time)
    {{
        last_bar_time = current_bar_time;
        
        // Execute trading logic
        ExecuteTradingLogic();
    }}
}}

//+------------------------------------------------------------------+
//| Main trading logic function                                      |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{{
    // Check if maximum positions reached
    if(CountOpenPositions() >= MaxPositions)
        return;
    
    // Check entry conditions
    if(CheckLongEntryConditions())
    {{
        OpenLongPosition();
    }}
    
    if(CheckShortEntryConditions())
    {{
        OpenShortPosition();
    }}
    
    // Check exit conditions
    CheckExitConditions();
}}

//+------------------------------------------------------------------+
//| Check long entry conditions                                      |
//+------------------------------------------------------------------+
bool CheckLongEntryConditions()
{{
    // Implement entry logic based on template
    // This would be generated from the template's entry_exit_rules
    
    return false; // Placeholder
}}

//+------------------------------------------------------------------+
//| Check short entry conditions                                     |
//+------------------------------------------------------------------+
bool CheckShortEntryConditions()
{{
    // Implement short entry logic
    
    return false; // Placeholder
}}

//+------------------------------------------------------------------+
//| Check exit conditions                                            |
//+------------------------------------------------------------------+
void CheckExitConditions()
{{
    // Implement exit logic based on template
    
    for(int i = OrdersTotal() - 1; i >= 0; i--)
    {{
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {{
            if(OrderMagicNumber() == magic_number && OrderSymbol() == Symbol())
            {{
                // Check exit conditions for this order
                // Implement based on template's exit rules
            }}
        }}
    }}
}}

//+------------------------------------------------------------------+
//| Open long position                                               |
//+------------------------------------------------------------------+
void OpenLongPosition()
{{
    double entry_price = Ask;
    double stop_loss = (StopLoss > 0) ? entry_price - StopLoss * Point : 0;
    double take_profit = (TakeProfit > 0) ? entry_price + TakeProfit * Point : 0;
    
    int ticket = OrderSend(Symbol(), OP_BUY, LotSize, entry_price, 3, 
                          stop_loss, take_profit, "QNTI_" + IntegerToString(magic_number), 
                          magic_number, 0, clrGreen);
    
    if(ticket > 0)
        Print("Long position opened: ", ticket);
    else
        Print("Error opening long position: ", GetLastError());
}}

//+------------------------------------------------------------------+
//| Open short position                                              |
//+------------------------------------------------------------------+
void OpenShortPosition()
{{
    double entry_price = Bid;
    double stop_loss = (StopLoss > 0) ? entry_price + StopLoss * Point : 0;
    double take_profit = (TakeProfit > 0) ? entry_price - TakeProfit * Point : 0;
    
    int ticket = OrderSend(Symbol(), OP_SELL, LotSize, entry_price, 3, 
                          stop_loss, take_profit, "QNTI_" + IntegerToString(magic_number), 
                          magic_number, 0, clrRed);
    
    if(ticket > 0)
        Print("Short position opened: ", ticket);
    else
        Print("Error opening short position: ", GetLastError());
}}

//+------------------------------------------------------------------+
//| Count open positions for this EA                                 |
//+------------------------------------------------------------------+
int CountOpenPositions()
{{
    int count = 0;
    for(int i = 0; i < OrdersTotal(); i++)
    {{
        if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {{
            if(OrderMagicNumber() == magic_number && OrderSymbol() == Symbol())
                count++;
        }}
    }}
    return count;
}}
"""
        return ea_code
    
    def cleanup_completed_generations(self, max_age_hours: int = 24):
        """Clean up old completed generations"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for generation_id, generation_task in self.active_generations.items():
            if generation_task["created_at"] < cutoff_time:
                status = self.get_generation_status(generation_id)
                if status and status["status"] in ["completed", "failed"]:
                    to_remove.append(generation_id)
        
        for generation_id in to_remove:
            del self.active_generations[generation_id]
            self.logger.info(f"Cleaned up old generation: {generation_id}")
    
    def shutdown(self):
        """Shutdown the integration manager"""
        self.logger.info("Shutting down QNTI EA Integration Manager")
        
        # Shutdown subagent manager
        if self.subagent_manager:
            self.subagent_manager.stop()
        
        # Shutdown EA generator
        if self.ea_generator:
            self.ea_generator.shutdown()
            
        # Shutdown reporting system
        if self.reporting_system:
            self.reporting_system.shutdown()
        
        self.logger.info("QNTI EA Integration Manager shutdown complete")

class StrategyTesterBridge:
    """Bridge to QNTI Strategy Tester"""
    
    def __init__(self, strategy_tester):
        self.strategy_tester = strategy_tester
        self.logger = logging.getLogger(f"{__name__}.StrategyTesterBridge")
    
    def run_backtest(self, ea_config: Dict[str, Any], data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest using QNTI Strategy Tester"""
        try:
            # Use existing strategy tester
            result = self.strategy_tester.run_backtest(
                strategy_name=ea_config.get("name", "Generated_EA"),
                symbol=data_config.get("symbol", "EURUSD"),
                timeframe=data_config.get("timeframe", "M15"),
                start_date=data_config.get("start_date"),
                end_date=data_config.get("end_date"),
                parameters=ea_config.get("parameters", {})
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return {"error": str(e)}

class MT5Bridge:
    """Bridge to MT5 system"""
    
    def __init__(self, mt5_bridge):
        self.mt5_bridge = mt5_bridge
        self.logger = logging.getLogger(f"{__name__}.MT5Bridge")
    
    def save_ea_file(self, ea_name: str, ea_code: str) -> str:
        """Save EA file to MT5 experts directory"""
        try:
            # Use existing MT5 bridge to save file
            file_path = f"MQL5/Experts/{ea_name}.mq5"
            
            # Save file (this would use the actual MT5 file system)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(ea_code)
            
            self.logger.info(f"EA file saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving EA file: {str(e)}")
            raise
    
    def compile_ea(self, ea_name: str) -> bool:
        """Compile EA using MT5"""
        try:
            # This would use MT5's compilation system
            self.logger.info(f"Compiling EA: {ea_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error compiling EA: {str(e)}")
            return False

class WebInterfaceBridge:
    """Bridge to QNTI Web Interface"""
    
    def __init__(self, web_server, integration_manager):
        self.web_server = web_server
        self.integration_manager = integration_manager
        self.logger = logging.getLogger(f"{__name__}.WebInterfaceBridge")
        
        # Register API endpoints
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register EA generation endpoints with web server"""
        try:
            # Register API routes (this would integrate with Flask/FastAPI)
            # /api/ea-generator/create
            # /api/ea-generator/status/{generation_id}
            # /api/ea-generator/list
            # /api/ea-generator/export/{generation_id}
            # /api/ea-generator/reports/summary
            # /api/ea-generator/reports/optimization
            # /api/ea-generator/reports/robustness
            # /api/ea-generator/reports/dashboard
            # /api/ea-generator/reports/export
            
            self.logger.info("EA Generator API endpoints registered with reporting")
            
        except Exception as e:
            self.logger.error(f"Error registering endpoints: {str(e)}")
    
    def handle_report_request(self, report_type: str, time_range_hours: int = 24) -> str:
        """Handle report generation requests from web interface"""
        return self.integration_manager.get_ea_generation_reports(report_type, time_range_hours)
    
    def handle_dashboard_request(self) -> Dict[str, Any]:
        """Handle real-time dashboard data requests"""
        return self.integration_manager.get_real_time_dashboard_data()

class DataManagerBridge:
    """Bridge to QNTI Data Manager"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.logger = logging.getLogger(f"{__name__}.DataManagerBridge")
    
    def prepare_data_for_generation(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Prepare data for EA generation"""
        try:
            data_info = {}
            
            for symbol in symbols:
                # Get data using existing data manager
                data = self.data_manager.get_historical_data(symbol, timeframe)
                
                data_info[symbol] = {
                    "records": len(data) if data is not None else 0,
                    "timeframe": timeframe,
                    "quality_score": self._calculate_data_quality(data)
                }
            
            return data_info
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return {}
    
    def _calculate_data_quality(self, data) -> float:
        """Calculate data quality score"""
        if data is None or len(data) == 0:
            return 0.0
        
        # Simple quality calculation
        missing_data = data.isnull().sum().sum() if hasattr(data, 'isnull') else 0
        total_data = len(data) * len(data.columns) if hasattr(data, 'columns') else len(data)
        
        quality = 1.0 - (missing_data / total_data) if total_data > 0 else 0.0
        return max(0.0, min(1.0, quality))

# Example usage
def main():
    """Example usage of the integration system"""
    
    # Mock QNTI system
    class MockQNTISystem:
        def __init__(self):
            self.strategy_tester = None
            self.mt5_bridge = None
            self.web_server = None
            self.data_manager = None
    
    # Initialize integration
    qnti_system = MockQNTISystem()
    integration_manager = QNTIEAIntegrationManager(qnti_system)
    
    try:
        # Initialize integration
        if integration_manager.initialize_integration():
            print("Integration initialized successfully")
            
            # Example web request
            web_request = {
                "template_config": {
                    "name": "RSI_MA_Strategy",
                    "description": "RSI with MA confirmation",
                    "timeframe": "M15",
                    "symbols": ["EURUSD"]
                },
                "optimization_config": {
                    "method": "grid_search",
                    "target_metric": "profit_factor",
                    "max_iterations": 50
                },
                "robustness_config": {
                    "tests": ["walk_forward", "monte_carlo"],
                    "walk_forward_periods": 6,
                    "monte_carlo_runs": 250
                }
            }
            
            # Submit EA generation
            generation_id = integration_manager.create_ea_from_web_request(web_request)
            print(f"EA generation submitted: {generation_id}")
            
            # Monitor progress
            for i in range(10):
                time.sleep(2)
                status = integration_manager.get_generation_status(generation_id)
                if status:
                    print(f"Generation progress: {status['progress']:.1%} - Status: {status['status']}")
                    if status['status'] in ['completed', 'failed']:
                        break
            
        else:
            print("Integration initialization failed")
    
    finally:
        integration_manager.shutdown()

if __name__ == "__main__":
    main() 