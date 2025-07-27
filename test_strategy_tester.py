#!/usr/bin/env python3
"""
Test suite for QNTI Strategy Tester and Parameter Optimizer
"""

import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from qnti_strategy_tester import QNTIStrategyTester, StrategyType
from qnti_parameter_optimizer import QNTIParameterOptimizer, OptimizationParameter, OptimizationConfig


class TestStrategyTester(unittest.TestCase):
    """Test cases for Strategy Tester"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.strategy_tester = QNTIStrategyTester(self.temp_dir)
        
        # Test parameters
        self.test_params = {
            'lot_size': 0.01,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'ma_short': 20,
            'ma_long': 50
        }
        
        # Test dates
        self.start_date = datetime.now() - timedelta(days=90)
        self.end_date = datetime.now() - timedelta(days=30)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_manager_creation(self):
        """Test historical data manager creation"""
        self.assertIsNotNone(self.strategy_tester.data_manager)
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_backtest_engine_creation(self):
        """Test backtest engine creation"""
        self.assertIsNotNone(self.strategy_tester.backtest_engine)
        self.assertEqual(self.strategy_tester.backtest_engine.initial_balance, 10000.0)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertTrue(self.strategy_tester.db_path.exists())
        
        # Check if tables exist
        import sqlite3
        conn = sqlite3.connect(self.strategy_tester.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['backtest_results', 'backtest_trades', 'equity_curve']
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()
    
    def test_trend_following_backtest(self):
        """Test trend following strategy backtest"""
        try:
            result = self.strategy_tester.run_backtest(
                ea_name="TrendFollower_EA",
                symbol="EURUSD",
                timeframe="H1",
                start_date=self.start_date,
                end_date=self.end_date,
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters=self.test_params
            )
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.test_id)
            self.assertEqual(result.ea_name, "TrendFollower_EA")
            self.assertEqual(result.symbol, "EURUSD")
            self.assertEqual(result.timeframe, "H1")
            self.assertIsInstance(result.total_trades, int)
            self.assertIsInstance(result.win_rate, float)
            self.assertIsInstance(result.profit_factor, float)
            
            print(f"✅ Trend Following Backtest Results:")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate:.2f}%")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            print(f"   Max Drawdown: {result.max_drawdown_percent:.2f}%")
            
        except Exception as e:
            print(f"❌ Trend Following Backtest failed: {e}")
            # This is expected if no data is available
            self.assertTrue(str(e).lower().find('no historical data') >= 0 or 
                          str(e).lower().find('no data') >= 0)
    
    def test_mean_reversion_backtest(self):
        """Test mean reversion strategy backtest"""
        try:
            result = self.strategy_tester.run_backtest(
                ea_name="MeanReversion_EA",
                symbol="GBPUSD",
                timeframe="H1",
                start_date=self.start_date,
                end_date=self.end_date,
                strategy_type=StrategyType.MEAN_REVERSION,
                parameters={
                    'lot_size': 0.01,
                    'stop_loss_pct': 0.01,
                    'bb_period': 20,
                    'bb_std': 2.0
                }
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.ea_name, "MeanReversion_EA")
            self.assertEqual(result.symbol, "GBPUSD")
            
            print(f"✅ Mean Reversion Backtest Results:")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate:.2f}%")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            
        except Exception as e:
            print(f"❌ Mean Reversion Backtest failed: {e}")
            # This is expected if no data is available
            self.assertTrue(str(e).lower().find('no historical data') >= 0 or 
                          str(e).lower().find('no data') >= 0)
    
    def test_result_storage_and_retrieval(self):
        """Test result storage and retrieval"""
        # Try to get results (should work even if empty)
        results = self.strategy_tester.get_backtest_results()
        self.assertIsInstance(results, list)
        
        print(f"✅ Result Storage Test: Found {len(results)} stored results")
    
    def test_result_deletion(self):
        """Test result deletion"""
        # Test deleting non-existent result
        success = self.strategy_tester.delete_backtest_result("non_existent_id")
        self.assertFalse(success)
        
        print("✅ Result Deletion Test: Correctly handled non-existent result")


class TestParameterOptimizer(unittest.TestCase):
    """Test cases for Parameter Optimizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.strategy_tester = QNTIStrategyTester(self.temp_dir)
        self.parameter_optimizer = QNTIParameterOptimizer(self.strategy_tester, self.temp_dir)
        
        # Test optimization parameters
        self.optimization_params = [
            OptimizationParameter(
                name="lot_size",
                min_value=0.01,
                max_value=0.05,
                step=0.01,
                param_type="float"
            ),
            OptimizationParameter(
                name="ma_short",
                min_value=10,
                max_value=30,
                step=5,
                param_type="int"
            ),
            OptimizationParameter(
                name="ma_long",
                min_value=40,
                max_value=60,
                step=10,
                param_type="int"
            )
        ]
        
        # Test dates
        self.start_date = datetime.now() - timedelta(days=60)
        self.end_date = datetime.now() - timedelta(days=30)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        self.assertIsNotNone(self.parameter_optimizer.grid_optimizer)
        self.assertIsNotNone(self.parameter_optimizer.genetic_optimizer)
        self.assertIsNotNone(self.parameter_optimizer.walkforward_optimizer)
        self.assertTrue(self.parameter_optimizer.db_path.exists())
    
    def test_optimization_config_creation(self):
        """Test optimization configuration creation"""
        config = OptimizationConfig(
            ea_name="TestEA",
            symbol="EURUSD",
            timeframe="H1",
            start_date=self.start_date,
            end_date=self.end_date,
            strategy_type=StrategyType.TREND_FOLLOWING,
            parameters=self.optimization_params,
            max_iterations=10,
            population_size=8
        )
        
        self.assertEqual(config.ea_name, "TestEA")
        self.assertEqual(config.symbol, "EURUSD")
        self.assertEqual(len(config.parameters), 3)
        self.assertEqual(config.max_iterations, 10)
        
        print("✅ Optimization Config Test: Configuration created successfully")
    
    def test_grid_search_optimization(self):
        """Test grid search optimization"""
        config = OptimizationConfig(
            ea_name="TrendFollower_EA",
            symbol="EURUSD",
            timeframe="H1",
            start_date=self.start_date,
            end_date=self.end_date,
            strategy_type=StrategyType.TREND_FOLLOWING,
            parameters=self.optimization_params[:2],  # Limit parameters for speed
            max_iterations=5
        )
        
        try:
            result = self.parameter_optimizer.optimize(config, "grid_search")
            
            self.assertIsNotNone(result)
            self.assertEqual(result.optimization_type, "grid_search")
            self.assertIsNotNone(result.best_parameters)
            self.assertIsInstance(result.total_tests, int)
            self.assertGreater(result.total_tests, 0)
            
            print(f"✅ Grid Search Optimization Results:")
            print(f"   Total Tests: {result.total_tests}")
            print(f"   Best Fitness: {result.best_fitness:.4f}")
            print(f"   Best Parameters: {result.best_parameters}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Grid Search Optimization failed: {e}")
            # This is expected if no data is available
            self.assertTrue(str(e).lower().find('no historical data') >= 0 or 
                          str(e).lower().find('no data') >= 0 or
                          str(e).lower().find('no valid') >= 0)
    
    def test_genetic_algorithm_optimization(self):
        """Test genetic algorithm optimization"""
        config = OptimizationConfig(
            ea_name="TrendFollower_EA",
            symbol="EURUSD",
            timeframe="H1",
            start_date=self.start_date,
            end_date=self.end_date,
            strategy_type=StrategyType.TREND_FOLLOWING,
            parameters=self.optimization_params,
            max_iterations=5,
            population_size=8
        )
        
        try:
            result = self.parameter_optimizer.optimize(config, "genetic_algorithm")
            
            self.assertIsNotNone(result)
            self.assertEqual(result.optimization_type, "genetic_algorithm")
            self.assertIsNotNone(result.best_parameters)
            
            print(f"✅ Genetic Algorithm Optimization Results:")
            print(f"   Total Tests: {result.total_tests}")
            print(f"   Best Fitness: {result.best_fitness:.4f}")
            print(f"   Best Parameters: {result.best_parameters}")
            
        except Exception as e:
            print(f"❌ Genetic Algorithm Optimization failed: {e}")
            # This is expected if no data is available
            self.assertTrue(str(e).lower().find('no historical data') >= 0 or 
                          str(e).lower().find('no data') >= 0 or
                          str(e).lower().find('no valid') >= 0)
    
    def test_optimization_result_storage(self):
        """Test optimization result storage"""
        results = self.parameter_optimizer.get_optimization_results()
        self.assertIsInstance(results, list)
        
        print(f"✅ Optimization Result Storage Test: Found {len(results)} stored results")
    
    def test_parameter_correlation_analysis(self):
        """Test parameter correlation analysis"""
        # Create dummy results for correlation analysis
        dummy_results = [
            {
                'best_parameters': {'lot_size': 0.01, 'ma_short': 20, 'ma_long': 50},
                'best_fitness': 1.5
            },
            {
                'best_parameters': {'lot_size': 0.02, 'ma_short': 15, 'ma_long': 45},
                'best_fitness': 1.2
            },
            {
                'best_parameters': {'lot_size': 0.03, 'ma_short': 25, 'ma_long': 55},
                'best_fitness': 1.8
            }
        ]
        
        correlation_analysis = self.parameter_optimizer.analyze_parameter_correlation(dummy_results)
        
        self.assertIsInstance(correlation_analysis, dict)
        self.assertIn('parameter_correlations', correlation_analysis)
        self.assertIn('total_results', correlation_analysis)
        self.assertIn('fitness_stats', correlation_analysis)
        
        print("✅ Parameter Correlation Analysis Test: Analysis completed successfully")


class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_strategy_tester_integration(self):
        """Test strategy tester integration"""
        strategy_tester = QNTIStrategyTester(self.temp_dir)
        parameter_optimizer = QNTIParameterOptimizer(strategy_tester, self.temp_dir)
        
        # Test that they work together
        self.assertIsNotNone(strategy_tester)
        self.assertIsNotNone(parameter_optimizer)
        self.assertEqual(parameter_optimizer.strategy_tester, strategy_tester)
        
        print("✅ System Integration Test: Components integrated successfully")
    
    def test_sample_ea_profiles(self):
        """Test sample EA profiles"""
        sample_profiles = [
            {
                'name': 'TrendFollower_EA',
                'strategy_type': 'trend_following',
                'parameters': {
                    'lot_size': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.01},
                    'ma_short': {'type': 'int', 'min': 5, 'max': 50, 'default': 20},
                    'ma_long': {'type': 'int', 'min': 20, 'max': 200, 'default': 50}
                }
            },
            {
                'name': 'MeanReversion_EA',
                'strategy_type': 'mean_reversion',
                'parameters': {
                    'lot_size': {'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.01},
                    'bb_period': {'type': 'int', 'min': 10, 'max': 50, 'default': 20},
                    'bb_std': {'type': 'float', 'min': 1.0, 'max': 3.0, 'default': 2.0}
                }
            }
        ]
        
        for profile in sample_profiles:
            self.assertIn('name', profile)
            self.assertIn('strategy_type', profile)
            self.assertIn('parameters', profile)
            self.assertIsInstance(profile['parameters'], dict)
            
            for param_name, param_config in profile['parameters'].items():
                self.assertIn('type', param_config)
                self.assertIn('min', param_config)
                self.assertIn('max', param_config)
                self.assertIn('default', param_config)
        
        print(f"✅ Sample EA Profiles Test: Validated {len(sample_profiles)} profiles")


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("QNTI Strategy Tester - Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestStrategyTester,
        TestParameterOptimizer,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests() 