"""
QNTI EA Generation Subagents
===========================

Distributed subagent system for parallel EA generation, optimization,
and robustness testing. Each subagent specializes in specific tasks
and can operate independently or in coordination.
"""

import asyncio
import logging
import json
import uuid
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubagentType(Enum):
    """Types of subagents"""
    OPTIMIZER = "optimizer"
    ROBUSTNESS_TESTER = "robustness_tester"
    PERFORMANCE_EVALUATOR = "performance_evaluator"
    DATA_MANAGER = "data_manager"
    TEMPLATE_BUILDER = "template_builder"
    RESULT_VALIDATOR = "result_validator"
    DEPLOYMENT_MANAGER = "deployment_manager"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SubagentTask:
    """Individual task for subagents"""
    task_id: str
    task_type: str
    subagent_type: SubagentType
    priority: TaskPriority
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    assigned_agent: Optional[str] = None

@dataclass
class SubagentConfig:
    """Configuration for subagents"""
    agent_id: str
    agent_type: SubagentType
    max_concurrent_tasks: int = 3
    specializations: List[str] = field(default_factory=list)
    performance_weight: float = 1.0
    enabled: bool = True

class BaseSubagent:
    """Base class for all subagents"""
    
    def __init__(self, config: SubagentConfig):
        self.config = config
        self.task_queue = queue.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.logger = logging.getLogger(f"{__name__}.{config.agent_id}")
        self.is_running = False
        self.worker_thread = None
        self.performance_stats = {
            "tasks_completed": 0,
            "average_execution_time": 0.0,
            "success_rate": 1.0,
            "last_active": datetime.now()
        }
    
    def start(self):
        """Start the subagent"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(f"Subagent {self.config.agent_id} started")
    
    def stop(self):
        """Stop the subagent"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        self.logger.info(f"Subagent {self.config.agent_id} stopped")
    
    def add_task(self, task: SubagentTask) -> bool:
        """Add task to queue"""
        if not self.config.enabled:
            return False
        
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            return False
        
        task.assigned_agent = self.config.agent_id
        self.task_queue.put(task)
        self.logger.info(f"Task {task.task_id} added to queue")
        return True
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                self._execute_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
    
    def _execute_task(self, task: SubagentTask):
        """Execute a single task"""
        start_time = time.time()
        
        try:
            task.status = TaskStatus.RUNNING
            self.active_tasks[task.task_id] = task
            
            self.logger.info(f"Executing task {task.task_id} ({task.task_type})")
            
            # Execute the specific task
            result = self._process_task(task)
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.execution_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(task, success=True)
            
            self.logger.info(f"Task {task.task_id} completed in {task.execution_time:.2f}s")
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.execution_time = time.time() - start_time
            
            self._update_performance_stats(task, success=False)
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
        
        finally:
            # Move task from active to completed
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
    
    def _process_task(self, task: SubagentTask) -> Any:
        """Process task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _process_task")
    
    def _update_performance_stats(self, task: SubagentTask, success: bool):
        """Update performance statistics"""
        self.performance_stats["tasks_completed"] += 1
        self.performance_stats["last_active"] = datetime.now()
        
        # Update average execution time
        current_avg = self.performance_stats["average_execution_time"]
        task_count = self.performance_stats["tasks_completed"]
        
        self.performance_stats["average_execution_time"] = (
            (current_avg * (task_count - 1) + task.execution_time) / task_count
        )
        
        # Update success rate
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for t in self.completed_tasks.values() 
                             if t.status == TaskStatus.COMPLETED)
        
        self.performance_stats["success_rate"] = successful_tasks / total_tasks if total_tasks > 0 else 1.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type.value,
            "is_running": self.is_running,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "performance_stats": self.performance_stats.copy()
        }

class OptimizerSubagent(BaseSubagent):
    """Subagent specialized in parameter optimization"""
    
    def _process_task(self, task: SubagentTask) -> Any:
        """Process optimization task"""
        task_type = task.parameters.get("optimization_type", "grid_search")
        
        if task_type == "grid_search":
            return self._grid_search_optimization(task)
        elif task_type == "genetic_algorithm":
            return self._genetic_algorithm_optimization(task)
        elif task_type == "differential_evolution":
            return self._differential_evolution_optimization(task)
        elif task_type == "bayesian_optimization":
            return self._bayesian_optimization(task)
        else:
            raise ValueError(f"Unknown optimization type: {task_type}")
    
    def _grid_search_optimization(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform grid search optimization"""
        parameters = task.parameters
        parameter_grid = parameters.get("parameter_grid", {})
        objective_function = parameters.get("objective_function")
        
        best_score = float('-inf')
        best_params = None
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # Evaluate this parameter set
            score = self._evaluate_parameters(param_dict, objective_function)
            
            results.append({
                "parameters": param_dict,
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_params = param_dict
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": results,
            "optimization_type": "grid_search"
        }
    
    def _genetic_algorithm_optimization(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform genetic algorithm optimization"""
        # Simplified GA implementation
        parameters = task.parameters
        population_size = parameters.get("population_size", 50)
        generations = parameters.get("generations", 100)
        mutation_rate = parameters.get("mutation_rate", 0.1)
        
        # This would implement a full genetic algorithm
        # For now, returning simulated results
        
        return {
            "best_parameters": {"param1": 1.5, "param2": 0.8},
            "best_score": 2.34,
            "generations_completed": generations,
            "optimization_type": "genetic_algorithm"
        }
    
    def _differential_evolution_optimization(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform differential evolution optimization"""
        from scipy.optimize import differential_evolution
        
        parameters = task.parameters
        bounds = parameters.get("bounds", [])
        objective_function = parameters.get("objective_function")
        
        def objective(params):
            param_dict = self._array_to_param_dict(params, parameters.get("param_names", []))
            return -self._evaluate_parameters(param_dict, objective_function)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=parameters.get("maxiter", 100),
            popsize=parameters.get("popsize", 15)
        )
        
        best_params = self._array_to_param_dict(result.x, parameters.get("param_names", []))
        
        return {
            "best_parameters": best_params,
            "best_score": -result.fun,
            "success": result.success,
            "function_evaluations": result.nfev,
            "optimization_type": "differential_evolution"
        }
    
    def _bayesian_optimization(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform Bayesian optimization"""
        # This would implement Bayesian optimization
        # For now, returning simulated results
        
        return {
            "best_parameters": {"param1": 1.2, "param2": 0.9},
            "best_score": 2.56,
            "acquisition_function": "expected_improvement",
            "optimization_type": "bayesian_optimization"
        }
    
    def _evaluate_parameters(self, param_dict: Dict[str, Any], objective_function: Optional[str]) -> float:
        """Evaluate parameter set using objective function"""
        # This would call the actual objective function
        # For now, returning a simulated score
        return np.random.uniform(0.5, 3.0)
    
    def _array_to_param_dict(self, params_array: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
        """Convert parameter array to dictionary"""
        return dict(zip(param_names, params_array))

class RobustnessSubagent(BaseSubagent):
    """Subagent specialized in robustness testing"""
    
    def _process_task(self, task: SubagentTask) -> Any:
        """Process robustness testing task"""
        test_type = task.parameters.get("test_type")
        
        if test_type == "walk_forward":
            return self._walk_forward_analysis(task)
        elif test_type == "monte_carlo":
            return self._monte_carlo_simulation(task)
        elif test_type == "parameter_sensitivity":
            return self._parameter_sensitivity_analysis(task)
        elif test_type == "stress_testing":
            return self._stress_testing(task)
        elif test_type == "out_of_sample":
            return self._out_of_sample_validation(task)
        else:
            raise ValueError(f"Unknown robustness test type: {test_type}")
    
    def _walk_forward_analysis(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform walk-forward analysis"""
        parameters = task.parameters
        
        # Simulate walk-forward analysis
        periods = parameters.get("periods", 12)
        results = []
        
        for i in range(periods):
            # Simulate performance for each period
            performance = np.random.uniform(0.8, 1.5)
            results.append({
                "period": i + 1,
                "performance": performance,
                "trades": np.random.randint(10, 50)
            })
        
        average_performance = np.mean([r["performance"] for r in results])
        stability = 1.0 - np.std([r["performance"] for r in results])
        
        return {
            "test_type": "walk_forward",
            "periods_tested": periods,
            "results": results,
            "average_performance": average_performance,
            "performance_stability": stability,
            "passed": stability > 0.7
        }
    
    def _monte_carlo_simulation(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform Monte Carlo simulation"""
        parameters = task.parameters
        runs = parameters.get("runs", 1000)
        
        # Simulate Monte Carlo runs
        results = []
        for i in range(runs):
            # Simulate randomized performance
            performance = np.random.normal(1.2, 0.3)
            results.append(performance)
        
        return {
            "test_type": "monte_carlo",
            "runs_completed": runs,
            "average_return": np.mean(results),
            "std_deviation": np.std(results),
            "worst_case": np.percentile(results, 5),
            "best_case": np.percentile(results, 95),
            "success_rate": len([r for r in results if r > 1.0]) / len(results),
            "passed": np.mean(results) > 1.1 and np.std(results) < 0.5
        }
    
    def _parameter_sensitivity_analysis(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform parameter sensitivity analysis"""
        parameters = task.parameters
        base_params = parameters.get("base_parameters", {})
        sensitivity_range = parameters.get("sensitivity_range", 0.1)
        
        sensitivity_results = {}
        
        for param_name, base_value in base_params.items():
            variations = []
            
            # Test parameter variations
            for variation in [-sensitivity_range, -sensitivity_range/2, 0, sensitivity_range/2, sensitivity_range]:
                if isinstance(base_value, (int, float)):
                    test_value = base_value * (1 + variation)
                    performance = np.random.uniform(0.9, 1.3)  # Simulate performance
                    variations.append({
                        "variation": variation,
                        "value": test_value,
                        "performance": performance
                    })
            
            # Calculate sensitivity score
            performances = [v["performance"] for v in variations]
            sensitivity_score = np.std(performances) / np.mean(performances)
            
            sensitivity_results[param_name] = {
                "variations": variations,
                "sensitivity_score": sensitivity_score,
                "robust": sensitivity_score < 0.2
            }
        
        overall_robustness = np.mean([r["sensitivity_score"] for r in sensitivity_results.values()])
        
        return {
            "test_type": "parameter_sensitivity",
            "parameter_results": sensitivity_results,
            "overall_robustness": overall_robustness,
            "passed": overall_robustness < 0.25
        }
    
    def _stress_testing(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform stress testing"""
        parameters = task.parameters
        
        # Simulate various stress conditions
        stress_results = {
            "high_volatility": np.random.uniform(0.6, 1.0),
            "low_volatility": np.random.uniform(0.8, 1.2),
            "trending_market": np.random.uniform(1.0, 1.4),
            "ranging_market": np.random.uniform(0.7, 1.1),
            "high_spread": np.random.uniform(0.5, 0.9),
            "slippage_impact": np.random.uniform(0.8, 1.0)
        }
        
        # Calculate overall stress resistance
        stress_scores = list(stress_results.values())
        average_performance = np.mean(stress_scores)
        worst_performance = min(stress_scores)
        
        return {
            "test_type": "stress_testing",
            "stress_conditions": stress_results,
            "average_performance": average_performance,
            "worst_performance": worst_performance,
            "stress_resistance": worst_performance / average_performance,
            "passed": worst_performance > 0.7 and average_performance > 0.85
        }
    
    def _out_of_sample_validation(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform out-of-sample validation"""
        parameters = task.parameters
        
        # Simulate in-sample and out-of-sample performance
        in_sample_performance = np.random.uniform(1.2, 1.8)
        out_of_sample_performance = np.random.uniform(0.8, 1.4)
        
        degradation = (in_sample_performance - out_of_sample_performance) / in_sample_performance
        
        return {
            "test_type": "out_of_sample",
            "in_sample_performance": in_sample_performance,
            "out_of_sample_performance": out_of_sample_performance,
            "performance_degradation": degradation,
            "overfitting_detected": degradation > 0.3,
            "passed": degradation < 0.25 and out_of_sample_performance > 1.0
        }

class PerformanceEvaluatorSubagent(BaseSubagent):
    """Subagent specialized in performance evaluation"""
    
    def _process_task(self, task: SubagentTask) -> Any:
        """Process performance evaluation task"""
        evaluation_type = task.parameters.get("evaluation_type", "comprehensive")
        
        if evaluation_type == "comprehensive":
            return self._comprehensive_evaluation(task)
        elif evaluation_type == "risk_metrics":
            return self._risk_metrics_evaluation(task)
        elif evaluation_type == "statistical_analysis":
            return self._statistical_analysis(task)
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    def _comprehensive_evaluation(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform comprehensive performance evaluation"""
        # Simulate comprehensive metrics
        return {
            "total_return": np.random.uniform(0.15, 0.35),
            "annual_return": np.random.uniform(0.12, 0.28),
            "sharpe_ratio": np.random.uniform(1.2, 2.0),
            "sortino_ratio": np.random.uniform(1.5, 2.5),
            "max_drawdown": np.random.uniform(0.05, 0.15),
            "profit_factor": np.random.uniform(1.3, 2.2),
            "win_rate": np.random.uniform(0.55, 0.75),
            "total_trades": np.random.randint(100, 500),
            "average_trade": np.random.uniform(0.001, 0.005),
            "largest_win": np.random.uniform(0.02, 0.05),
            "largest_loss": np.random.uniform(-0.03, -0.01),
            "recovery_factor": np.random.uniform(2.0, 5.0),
            "calmar_ratio": np.random.uniform(1.5, 3.0),
            "sterling_ratio": np.random.uniform(1.2, 2.5)
        }
    
    def _risk_metrics_evaluation(self, task: SubagentTask) -> Dict[str, Any]:
        """Evaluate risk-specific metrics"""
        return {
            "value_at_risk_95": np.random.uniform(-0.03, -0.01),
            "conditional_var_95": np.random.uniform(-0.04, -0.02),
            "maximum_consecutive_losses": np.random.randint(3, 8),
            "drawdown_duration": np.random.randint(5, 20),
            "volatility": np.random.uniform(0.15, 0.35),
            "beta": np.random.uniform(0.7, 1.3),
            "tracking_error": np.random.uniform(0.02, 0.08)
        }
    
    def _statistical_analysis(self, task: SubagentTask) -> Dict[str, Any]:
        """Perform statistical analysis"""
        return {
            "trade_distribution": "normal",
            "autocorrelation": np.random.uniform(-0.1, 0.1),
            "kurtosis": np.random.uniform(0.5, 3.0),
            "skewness": np.random.uniform(-0.5, 0.5),
            "jarque_bera_test": {"statistic": 2.45, "p_value": 0.29},
            "ljung_box_test": {"statistic": 5.67, "p_value": 0.84}
        }

class DataManagerSubagent(BaseSubagent):
    """Subagent specialized in data management"""
    
    def _process_task(self, task: SubagentTask) -> Any:
        """Process data management task"""
        task_type = task.parameters.get("task_type")
        
        if task_type == "data_preparation":
            return self._prepare_data(task)
        elif task_type == "data_validation":
            return self._validate_data(task)
        elif task_type == "data_splitting":
            return self._split_data(task)
        else:
            raise ValueError(f"Unknown data task type: {task_type}")
    
    def _prepare_data(self, task: SubagentTask) -> Dict[str, Any]:
        """Prepare data for backtesting"""
        parameters = task.parameters
        
        # Simulate data preparation
        return {
            "data_points": np.random.randint(10000, 50000),
            "timeframe": parameters.get("timeframe", "M15"),
            "symbols": parameters.get("symbols", ["EURUSD"]),
            "preparation_time": np.random.uniform(0.5, 2.0),
            "quality_score": np.random.uniform(0.85, 0.99)
        }
    
    def _validate_data(self, task: SubagentTask) -> Dict[str, Any]:
        """Validate data quality"""
        return {
            "missing_data_percentage": np.random.uniform(0.0, 2.0),
            "outliers_detected": np.random.randint(0, 10),
            "data_consistency": np.random.uniform(0.95, 1.0),
            "validation_passed": True
        }
    
    def _split_data(self, task: SubagentTask) -> Dict[str, Any]:
        """Split data for training/testing"""
        parameters = task.parameters
        split_ratio = parameters.get("split_ratio", 0.8)
        
        return {
            "train_size": int(10000 * split_ratio),
            "test_size": int(10000 * (1 - split_ratio)),
            "validation_size": int(10000 * 0.1),
            "split_method": parameters.get("split_method", "chronological")
        }

class SubagentManager:
    """Manages all subagents and task distribution"""
    
    def __init__(self):
        self.subagents = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = {}
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.dispatcher_thread = None
        
        # Initialize default subagents
        self._initialize_default_subagents()
    
    def _initialize_default_subagents(self):
        """Initialize default set of subagents"""
        # Optimizer subagents
        for i in range(2):
            config = SubagentConfig(
                agent_id=f"optimizer_{i}",
                agent_type=SubagentType.OPTIMIZER,
                max_concurrent_tasks=3,
                specializations=["grid_search", "genetic_algorithm"]
            )
            self.subagents[config.agent_id] = OptimizerSubagent(config)
        
        # Robustness testing subagents
        for i in range(2):
            config = SubagentConfig(
                agent_id=f"robustness_{i}",
                agent_type=SubagentType.ROBUSTNESS_TESTER,
                max_concurrent_tasks=2,
                specializations=["walk_forward", "monte_carlo"]
            )
            self.subagents[config.agent_id] = RobustnessSubagent(config)
        
        # Performance evaluator subagent
        config = SubagentConfig(
            agent_id="performance_evaluator",
            agent_type=SubagentType.PERFORMANCE_EVALUATOR,
            max_concurrent_tasks=4
        )
        self.subagents[config.agent_id] = PerformanceEvaluatorSubagent(config)
        
        # Data manager subagent
        config = SubagentConfig(
            agent_id="data_manager",
            agent_type=SubagentType.DATA_MANAGER,
            max_concurrent_tasks=5
        )
        self.subagents[config.agent_id] = DataManagerSubagent(config)
    
    def start(self):
        """Start the subagent manager"""
        if self.is_running:
            return
        
        # Start all subagents
        for agent in self.subagents.values():
            agent.start()
        
        # Start task dispatcher
        self.is_running = True
        self.dispatcher_thread = threading.Thread(target=self._dispatch_tasks, daemon=True)
        self.dispatcher_thread.start()
        
        self.logger.info("Subagent Manager started")
    
    def stop(self):
        """Stop the subagent manager"""
        self.is_running = False
        
        # Stop all subagents
        for agent in self.subagents.values():
            agent.stop()
        
        # Stop dispatcher
        if self.dispatcher_thread:
            self.dispatcher_thread.join(timeout=5.0)
        
        self.logger.info("Subagent Manager stopped")
    
    def submit_task(self, task: SubagentTask) -> str:
        """Submit task for execution"""
        # Add priority for queue ordering (lower number = higher priority)
        priority = 5 - task.priority.value
        self.task_queue.put((priority, task.created_at, task))
        
        self.logger.info(f"Task {task.task_id} submitted with priority {task.priority.value}")
        return task.task_id
    
    def _dispatch_tasks(self):
        """Dispatch tasks to appropriate subagents"""
        while self.is_running:
            try:
                # Get task from priority queue
                priority, created_at, task = self.task_queue.get(timeout=1.0)
                
                # Find suitable subagent
                suitable_agents = self._find_suitable_agents(task)
                
                if suitable_agents:
                    # Select best agent based on performance and availability
                    best_agent = self._select_best_agent(suitable_agents)
                    
                    # Assign task to agent
                    if best_agent and best_agent.add_task(task):
                        self.logger.info(f"Task {task.task_id} assigned to {best_agent.config.agent_id}")
                    else:
                        # Agent couldn't accept task, put it back in queue
                        self.task_queue.put((priority, created_at, task))
                else:
                    # No suitable agents available, put task back
                    self.task_queue.put((priority, created_at, task))
                    time.sleep(1.0)  # Wait before retrying
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error dispatching tasks: {str(e)}")
    
    def _find_suitable_agents(self, task: SubagentTask) -> List[BaseSubagent]:
        """Find subagents suitable for the task"""
        suitable = []
        
        for agent in self.subagents.values():
            if (agent.config.agent_type == task.subagent_type and 
                agent.config.enabled and 
                len(agent.active_tasks) < agent.config.max_concurrent_tasks):
                suitable.append(agent)
        
        return suitable
    
    def _select_best_agent(self, agents: List[BaseSubagent]) -> Optional[BaseSubagent]:
        """Select the best agent based on performance and load"""
        if not agents:
            return None
        
        best_agent = None
        best_score = float('-inf')
        
        for agent in agents:
            # Calculate selection score based on performance and load
            load_factor = len(agent.active_tasks) / agent.config.max_concurrent_tasks
            performance_factor = agent.performance_stats["success_rate"]
            speed_factor = 1.0 / (agent.performance_stats["average_execution_time"] + 0.1)
            
            score = (performance_factor * 0.4 + 
                    (1 - load_factor) * 0.4 + 
                    speed_factor * 0.2)
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks in all agents
        for agent in self.subagents.values():
            if task_id in agent.active_tasks:
                task = agent.active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent,
                    "execution_time": task.execution_time
                }
            
            if task_id in agent.completed_tasks:
                task = agent.completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent,
                    "execution_time": task.execution_time,
                    "result": task.result,
                    "error": task.error
                }
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_active_tasks = sum(len(agent.active_tasks) for agent in self.subagents.values())
        total_completed_tasks = sum(len(agent.completed_tasks) for agent in self.subagents.values())
        
        agent_statuses = {}
        for agent_id, agent in self.subagents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        return {
            "is_running": self.is_running,
            "total_agents": len(self.subagents),
            "queue_size": self.task_queue.qsize(),
            "total_active_tasks": total_active_tasks,
            "total_completed_tasks": total_completed_tasks,
            "agents": agent_statuses
        }
    
    def add_subagent(self, agent: BaseSubagent):
        """Add a new subagent"""
        self.subagents[agent.config.agent_id] = agent
        if self.is_running:
            agent.start()
        
        self.logger.info(f"Added subagent: {agent.config.agent_id}")
    
    def remove_subagent(self, agent_id: str):
        """Remove a subagent"""
        if agent_id in self.subagents:
            agent = self.subagents[agent_id]
            agent.stop()
            del self.subagents[agent_id]
            
            self.logger.info(f"Removed subagent: {agent_id}")

# Example usage
def main():
    """Example usage of the subagent system"""
    
    # Initialize subagent manager
    manager = SubagentManager()
    manager.start()
    
    try:
        # Submit some example tasks
        
        # Optimization task
        opt_task = SubagentTask(
            task_id=str(uuid.uuid4()),
            task_type="grid_search_optimization",
            subagent_type=SubagentType.OPTIMIZER,
            priority=TaskPriority.HIGH,
            parameters={
                "optimization_type": "grid_search",
                "parameter_grid": {
                    "rsi_period": [10, 14, 20],
                    "ma_period": [20, 50, 100]
                }
            }
        )
        
        task_id = manager.submit_task(opt_task)
        print(f"Submitted optimization task: {task_id}")
        
        # Robustness testing task
        robust_task = SubagentTask(
            task_id=str(uuid.uuid4()),
            task_type="walk_forward_analysis",
            subagent_type=SubagentType.ROBUSTNESS_TESTER,
            priority=TaskPriority.NORMAL,
            parameters={
                "test_type": "walk_forward",
                "periods": 12
            }
        )
        
        task_id = manager.submit_task(robust_task)
        print(f"Submitted robustness task: {task_id}")
        
        # Wait for tasks to complete
        time.sleep(5)
        
        # Check system status
        status = manager.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
    finally:
        manager.stop()

if __name__ == "__main__":
    main() 