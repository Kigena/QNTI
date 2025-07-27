#!/usr/bin/env python3
"""
QNTI Parameter Optimizer
Advanced parameter optimization for trading strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import sqlite3
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid

from qnti_strategy_tester import QNTIStrategyTester, StrategyType

logger = logging.getLogger(__name__)

@dataclass
class OptimizationParameter:
    """Parameter definition for optimization"""
    name: str
    min_value: float
    max_value: float
    step: float
    param_type: str = "float"  # float, int, bool

@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    ea_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    strategy_type: StrategyType
    parameters: List[OptimizationParameter]
    initial_balance: float = 10000.0
    fitness_function: str = "profit_factor"  # profit_factor, sharpe_ratio, total_return
    max_iterations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

@dataclass
class OptimizationResult:
    """Result of optimization run"""
    config_id: str
    ea_name: str
    symbol: str
    timeframe: str
    optimization_type: str
    best_parameters: Dict[str, Any]
    best_fitness: float
    total_tests: int
    execution_time: float
    created_at: datetime
    results_summary: Dict[str, Any]

class GridSearchOptimizer:
    """Grid search parameter optimization"""
    
    def __init__(self, strategy_tester: QNTIStrategyTester):
        self.strategy_tester = strategy_tester
        
    def optimize(self, config: OptimizationConfig) -> OptimizationResult:
        """Run grid search optimization"""
        logger.info(f"Starting grid search optimization for {config.ea_name}")
        
        start_time = time.time()
        config_id = str(uuid.uuid4())
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(config.parameters)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Test all combinations
        results = []
        
        # Use parallel processing for faster optimization
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for params in param_combinations:
                future = executor.submit(self._test_parameters, config, params)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in parameter test: {e}")
        
        # Find best result
        if not results:
            raise ValueError("No valid optimization results")
        
        best_result = max(results, key=lambda x: x.get(config.fitness_function, 0))
        
        optimization_result = OptimizationResult(
            config_id=config_id,
            ea_name=config.ea_name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            optimization_type="grid_search",
            best_parameters=best_result['parameters'],
            best_fitness=best_result.get(config.fitness_function, 0),
            total_tests=len(results),
            execution_time=time.time() - start_time,
            created_at=datetime.now(),
            results_summary={
                'total_combinations': len(param_combinations),
                'successful_tests': len(results),
                'best_metrics': {k: v for k, v in best_result.items() if k != 'parameters'}
            }
        )
        
        logger.info(f"Grid search completed in {optimization_result.execution_time:.2f}s")
        return optimization_result
    
    def _generate_parameter_combinations(self, parameters: List[OptimizationParameter]) -> List[Dict]:
        """Generate all parameter combinations for grid search"""
        combinations = [{}]
        
        for param in parameters:
            new_combinations = []
            
            if param.param_type == "float":
                values = np.arange(param.min_value, param.max_value + param.step, param.step)
            elif param.param_type == "int":
                values = range(int(param.min_value), int(param.max_value) + 1, int(param.step))
            elif param.param_type == "bool":
                values = [True, False]
            else:
                values = [param.min_value]
            
            for combo in combinations:
                for value in values:
                    new_combo = combo.copy()
                    new_combo[param.name] = value
                    new_combinations.append(new_combo)
            
            combinations = new_combinations
        
        return combinations
    
    def _test_parameters(self, config: OptimizationConfig, parameters: Dict) -> Optional[Dict]:
        """Test a specific parameter set"""
        try:
            result = self.strategy_tester.run_backtest(
                ea_name=config.ea_name,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                strategy_type=config.strategy_type,
                parameters=parameters,
                initial_balance=config.initial_balance
            )
            
            return {
                'parameters': parameters,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': ((result.final_balance - result.initial_balance) / result.initial_balance) * 100,
                'max_drawdown_percent': result.max_drawdown_percent,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            }
            
        except Exception as e:
            logger.error(f"Error testing parameters {parameters}: {e}")
            return None

class GeneticAlgorithmOptimizer:
    """Genetic algorithm parameter optimization"""
    
    def __init__(self, strategy_tester: QNTIStrategyTester):
        self.strategy_tester = strategy_tester
        
    def optimize(self, config: OptimizationConfig) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        logger.info(f"Starting genetic algorithm optimization for {config.ea_name}")
        
        start_time = time.time()
        config_id = str(uuid.uuid4())
        
        # Initialize population
        population = self._initialize_population(config)
        
        best_fitness = float('-inf')
        best_parameters = None
        total_tests = 0
        
        for generation in range(config.max_iterations):
            logger.info(f"Generation {generation + 1}/{config.max_iterations}")
            
            # Evaluate population
            fitness_scores = []
            for individual in population:
                try:
                    result = self._test_parameters(config, individual)
                    if result:
                        fitness = result.get(config.fitness_function, 0)
                        fitness_scores.append(fitness)
                        total_tests += 1
                        
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_parameters = individual.copy()
                    else:
                        fitness_scores.append(float('-inf'))
                except Exception as e:
                    logger.error(f"Error evaluating individual: {e}")
                    fitness_scores.append(float('-inf'))
            
            # Selection, crossover, and mutation
            population = self._evolve_population(population, fitness_scores, config)
            
            logger.info(f"Generation {generation + 1} best fitness: {best_fitness:.4f}")
        
        optimization_result = OptimizationResult(
            config_id=config_id,
            ea_name=config.ea_name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            optimization_type="genetic_algorithm",
            best_parameters=best_parameters,
            best_fitness=best_fitness,
            total_tests=total_tests,
            execution_time=time.time() - start_time,
            created_at=datetime.now(),
            results_summary={
                'generations': config.max_iterations,
                'population_size': config.population_size,
                'mutation_rate': config.mutation_rate,
                'crossover_rate': config.crossover_rate
            }
        )
        
        logger.info(f"Genetic algorithm completed in {optimization_result.execution_time:.2f}s")
        return optimization_result
    
    def _initialize_population(self, config: OptimizationConfig) -> List[Dict]:
        """Initialize random population"""
        population = []
        
        for _ in range(config.population_size):
            individual = {}
            for param in config.parameters:
                if param.param_type == "float":
                    value = random.uniform(param.min_value, param.max_value)
                elif param.param_type == "int":
                    value = random.randint(int(param.min_value), int(param.max_value))
                elif param.param_type == "bool":
                    value = random.choice([True, False])
                else:
                    value = param.min_value
                
                individual[param.name] = value
            
            population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[Dict], fitness_scores: List[float], 
                          config: OptimizationConfig) -> List[Dict]:
        """Evolve population using selection, crossover, and mutation"""
        
        # Selection (tournament selection)
        selected = []
        for _ in range(config.population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        
        # Crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover
            if random.random() < config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, config.parameters)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < config.mutation_rate:
                child1 = self._mutate(child1, config.parameters)
            if random.random() < config.mutation_rate:
                child2 = self._mutate(child2, config.parameters)
            
            new_population.extend([child1, child2])
        
        return new_population[:config.population_size]
    
    def _crossover(self, parent1: Dict, parent2: Dict, 
                  parameters: List[OptimizationParameter]) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parameters) - 1)
        param_names = [p.name for p in parameters]
        
        for i in range(crossover_point, len(param_names)):
            param_name = param_names[i]
            child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict, parameters: List[OptimizationParameter]) -> Dict:
        """Mutate an individual"""
        mutated = individual.copy()
        
        for param in parameters:
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                if param.param_type == "float":
                    # Gaussian mutation
                    current_value = mutated[param.name]
                    range_size = param.max_value - param.min_value
                    mutation = random.gauss(0, range_size * 0.1)
                    new_value = current_value + mutation
                    mutated[param.name] = max(param.min_value, min(param.max_value, new_value))
                elif param.param_type == "int":
                    mutated[param.name] = random.randint(int(param.min_value), int(param.max_value))
                elif param.param_type == "bool":
                    mutated[param.name] = not mutated[param.name]
        
        return mutated
    
    def _test_parameters(self, config: OptimizationConfig, parameters: Dict) -> Optional[Dict]:
        """Test a specific parameter set"""
        try:
            result = self.strategy_tester.run_backtest(
                ea_name=config.ea_name,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                strategy_type=config.strategy_type,
                parameters=parameters,
                initial_balance=config.initial_balance
            )
            
            return {
                'parameters': parameters,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': ((result.final_balance - result.initial_balance) / result.initial_balance) * 100,
                'max_drawdown_percent': result.max_drawdown_percent,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            }
            
        except Exception as e:
            logger.error(f"Error testing parameters {parameters}: {e}")
            return None

class WalkForwardOptimizer:
    """Walk-forward analysis optimizer"""
    
    def __init__(self, strategy_tester: QNTIStrategyTester):
        self.strategy_tester = strategy_tester
        
    def optimize(self, config: OptimizationConfig, 
                optimization_window_days: int = 180,
                test_window_days: int = 30) -> OptimizationResult:
        """Run walk-forward optimization"""
        logger.info(f"Starting walk-forward optimization for {config.ea_name}")
        
        start_time = time.time()
        config_id = str(uuid.uuid4())
        
        # Calculate walk-forward periods
        total_days = (config.end_date - config.start_date).days
        periods = []
        
        current_date = config.start_date
        while current_date + timedelta(days=optimization_window_days + test_window_days) <= config.end_date:
            opt_start = current_date
            opt_end = current_date + timedelta(days=optimization_window_days)
            test_start = opt_end
            test_end = opt_end + timedelta(days=test_window_days)
            
            periods.append({
                'opt_start': opt_start,
                'opt_end': opt_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_date = test_start
        
        logger.info(f"Walk-forward periods: {len(periods)}")
        
        # Run optimization for each period
        period_results = []
        total_tests = 0
        
        for i, period in enumerate(periods):
            logger.info(f"Walk-forward period {i+1}/{len(periods)}")
            
            # Create config for optimization period
            opt_config = OptimizationConfig(
                ea_name=config.ea_name,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=period['opt_start'],
                end_date=period['opt_end'],
                strategy_type=config.strategy_type,
                parameters=config.parameters,
                initial_balance=config.initial_balance,
                fitness_function=config.fitness_function,
                max_iterations=min(config.max_iterations, 100),  # Reduced for walk-forward
                population_size=min(config.population_size, 20)
            )
            
            # Run optimization
            optimizer = GeneticAlgorithmOptimizer(self.strategy_tester)
            opt_result = optimizer.optimize(opt_config)
            
            # Test optimized parameters on out-of-sample period
            test_result = self.strategy_tester.run_backtest(
                ea_name=config.ea_name,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=period['test_start'],
                end_date=period['test_end'],
                strategy_type=config.strategy_type,
                parameters=opt_result.best_parameters,
                initial_balance=config.initial_balance
            )
            
            period_results.append({
                'period': i + 1,
                'optimization_period': f"{period['opt_start'].strftime('%Y-%m-%d')} to {period['opt_end'].strftime('%Y-%m-%d')}",
                'test_period': f"{period['test_start'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}",
                'optimized_parameters': opt_result.best_parameters,
                'in_sample_fitness': opt_result.best_fitness,
                'out_of_sample_fitness': getattr(test_result, config.fitness_function, 0),
                'test_result': {
                    'profit_factor': test_result.profit_factor,
                    'sharpe_ratio': test_result.sharpe_ratio,
                    'total_return': ((test_result.final_balance - test_result.initial_balance) / test_result.initial_balance) * 100,
                    'max_drawdown_percent': test_result.max_drawdown_percent,
                    'win_rate': test_result.win_rate,
                    'total_trades': test_result.total_trades
                }
            })
            
            total_tests += opt_result.total_tests + 1
        
        # Calculate overall performance
        overall_fitness = np.mean([r['out_of_sample_fitness'] for r in period_results])
        
        # Use most recent optimized parameters as best
        best_parameters = period_results[-1]['optimized_parameters'] if period_results else {}
        
        optimization_result = OptimizationResult(
            config_id=config_id,
            ea_name=config.ea_name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            optimization_type="walk_forward",
            best_parameters=best_parameters,
            best_fitness=overall_fitness,
            total_tests=total_tests,
            execution_time=time.time() - start_time,
            created_at=datetime.now(),
            results_summary={
                'periods': len(periods),
                'optimization_window_days': optimization_window_days,
                'test_window_days': test_window_days,
                'period_results': period_results,
                'average_out_of_sample_fitness': overall_fitness,
                'fitness_stability': np.std([r['out_of_sample_fitness'] for r in period_results])
            }
        )
        
        logger.info(f"Walk-forward optimization completed in {optimization_result.execution_time:.2f}s")
        return optimization_result

class QNTIParameterOptimizer:
    """Main parameter optimizer class"""
    
    def __init__(self, strategy_tester: QNTIStrategyTester, data_dir: str = "qnti_data"):
        self.strategy_tester = strategy_tester
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize optimizers
        self.grid_optimizer = GridSearchOptimizer(strategy_tester)
        self.genetic_optimizer = GeneticAlgorithmOptimizer(strategy_tester)
        self.walkforward_optimizer = WalkForwardOptimizer(strategy_tester)
        
        # Database for storing results
        self.db_path = self.data_dir / "optimization_results.db"
        self._init_database()
        
        logger.info("QNTI Parameter Optimizer initialized")
    
    def _init_database(self):
        """Initialize database for optimization results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_results (
                config_id TEXT PRIMARY KEY,
                ea_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                optimization_type TEXT,
                best_parameters TEXT,
                best_fitness REAL,
                total_tests INTEGER,
                execution_time REAL,
                created_at TEXT,
                results_summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def optimize(self, config: OptimizationConfig, 
                optimization_type: str = "genetic_algorithm") -> OptimizationResult:
        """Run parameter optimization"""
        
        if optimization_type == "grid_search":
            result = self.grid_optimizer.optimize(config)
        elif optimization_type == "genetic_algorithm":
            result = self.genetic_optimizer.optimize(config)
        elif optimization_type == "walk_forward":
            result = self.walkforward_optimizer.optimize(config)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        # Save result
        self._save_optimization_result(result)
        
        return result
    
    def _save_optimization_result(self, result: OptimizationResult):
        """Save optimization result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO optimization_results (
                    config_id, ea_name, symbol, timeframe, optimization_type,
                    best_parameters, best_fitness, total_tests, execution_time,
                    created_at, results_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.config_id, result.ea_name, result.symbol, result.timeframe,
                result.optimization_type, json.dumps(result.best_parameters),
                result.best_fitness, result.total_tests, result.execution_time,
                result.created_at.isoformat(), json.dumps(result.results_summary)
            ))
            
            conn.commit()
            logger.info(f"Optimization result saved: {result.config_id}")
            
        except Exception as e:
            logger.error(f"Error saving optimization result: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_optimization_results(self, limit: int = 100) -> List[Dict]:
        """Get optimization results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM optimization_results
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            result = {
                'config_id': row[0],
                'ea_name': row[1],
                'symbol': row[2],
                'timeframe': row[3],
                'optimization_type': row[4],
                'best_parameters': json.loads(row[5]) if row[5] else {},
                'best_fitness': row[6],
                'total_tests': row[7],
                'execution_time': row[8],
                'created_at': row[9],
                'results_summary': json.loads(row[10]) if row[10] else {}
            }
            results.append(result)
        
        conn.close()
        return results
    
    def analyze_parameter_correlation(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze parameter correlation with performance"""
        if not results:
            return {}
        
        # Collect parameter values and fitness scores
        parameter_data = {}
        fitness_scores = []
        
        for result in results:
            fitness_scores.append(result['best_fitness'])
            
            for param_name, param_value in result['best_parameters'].items():
                if param_name not in parameter_data:
                    parameter_data[param_name] = []
                parameter_data[param_name].append(param_value)
        
        # Calculate correlations
        correlations = {}
        for param_name, values in parameter_data.items():
            if len(values) == len(fitness_scores):
                try:
                    # Only calculate for numeric parameters
                    if all(isinstance(v, (int, float)) for v in values):
                        correlation = np.corrcoef(values, fitness_scores)[0, 1]
                        correlations[param_name] = correlation
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {param_name}: {e}")
        
        return {
            'parameter_correlations': correlations,
            'total_results': len(results),
            'fitness_stats': {
                'mean': np.mean(fitness_scores),
                'std': np.std(fitness_scores),
                'min': np.min(fitness_scores),
                'max': np.max(fitness_scores)
            }
        }
    
    def generate_sensitivity_report(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate parameter sensitivity report"""
        logger.info("Generating parameter sensitivity report")
        
        base_parameters = {}
        for param in config.parameters:
            # Use middle value as base
            if param.param_type == "float":
                base_parameters[param.name] = (param.min_value + param.max_value) / 2
            elif param.param_type == "int":
                base_parameters[param.name] = int((param.min_value + param.max_value) / 2)
            elif param.param_type == "bool":
                base_parameters[param.name] = True
        
        # Test base parameters
        base_result = self.strategy_tester.run_backtest(
            ea_name=config.ea_name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date,
            strategy_type=config.strategy_type,
            parameters=base_parameters,
            initial_balance=config.initial_balance
        )
        
        base_fitness = getattr(base_result, config.fitness_function, 0)
        
        # Test sensitivity for each parameter
        sensitivity_results = {}
        
        for param in config.parameters:
            if param.param_type in ["float", "int"]:
                # Test parameter at different values
                test_values = []
                if param.param_type == "float":
                    test_values = [param.min_value, base_parameters[param.name], param.max_value]
                else:
                    test_values = [int(param.min_value), base_parameters[param.name], int(param.max_value)]
                
                param_results = []
                for value in test_values:
                    test_params = base_parameters.copy()
                    test_params[param.name] = value
                    
                    try:
                        result = self.strategy_tester.run_backtest(
                            ea_name=config.ea_name,
                            symbol=config.symbol,
                            timeframe=config.timeframe,
                            start_date=config.start_date,
                            end_date=config.end_date,
                            strategy_type=config.strategy_type,
                            parameters=test_params,
                            initial_balance=config.initial_balance
                        )
                        
                        fitness = getattr(result, config.fitness_function, 0)
                        param_results.append({
                            'value': value,
                            'fitness': fitness,
                            'fitness_change': fitness - base_fitness
                        })
                        
                    except Exception as e:
                        logger.error(f"Error testing parameter {param.name}={value}: {e}")
                
                sensitivity_results[param.name] = {
                    'parameter_type': param.param_type,
                    'test_results': param_results,
                    'sensitivity_score': max([abs(r['fitness_change']) for r in param_results]) if param_results else 0
                }
        
        return {
            'base_parameters': base_parameters,
            'base_fitness': base_fitness,
            'sensitivity_results': sensitivity_results,
            'most_sensitive_parameters': sorted(
                sensitivity_results.keys(),
                key=lambda x: sensitivity_results[x]['sensitivity_score'],
                reverse=True
            )[:3]
        }

# Example usage
if __name__ == "__main__":
    # Example optimization configuration
    from qnti_core_system import QNTITradeManager
    from qnti_strategy_tester import QNTIStrategyTester
    
    # Initialize components
    trade_manager = QNTITradeManager()
    strategy_tester = QNTIStrategyTester(trade_manager)
    
    optimizer = QNTIParameterOptimizer(strategy_tester)
    
    # Create optimization configuration
    config = OptimizationConfig(
        ea_name="GOLDEN STREAK",
        symbol="XAUUSD",
        timeframe="1H",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        strategy_type=StrategyType.LONG,
        parameters=[
            OptimizationParameter(
                name="risk_percent",
                min_value=0.01,
                max_value=0.05,
                step=0.005,
                param_type="float"
            ),
            OptimizationParameter(
                name="stop_loss_pips",
                min_value=20,
                max_value=100,
                step=10,
                param_type="int"
            ),
            OptimizationParameter(
                name="take_profit_pips",
                min_value=30,
                max_value=200,
                step=20,
                param_type="int"
            )
        ],
        initial_balance=10000.0,
        fitness_function="profit_factor",
        max_iterations=1000,
        population_size=50
    )
    
    # Run optimization
    result = optimizer.optimize(config)
    
    if result:
        print("Optimization completed!")
        print(f"Best parameters: {result.best_parameters}")
        print(f"Best fitness: {result.best_fitness}")
        
        # Generate report
        report = optimizer.create_optimization_report(result.optimization_id)
        print("\nOptimization Report:")
        print(json.dumps(report, indent=2, default=str)) 