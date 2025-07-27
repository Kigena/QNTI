#!/usr/bin/env python3
"""
QNTI EA Generation Comprehensive Reporting System
Advanced logging, analytics, and reporting for EA generation pipeline
"""

import logging
import json
import csv
import sqlite3
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: Plotting libraries not available. Install matplotlib, seaborn, plotly for full functionality.")

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Note: Jinja2 not available. Install jinja2 for HTML report generation.")

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
import time
from collections import defaultdict, deque

class ReportType(Enum):
    """Types of reports that can be generated"""
    GENERATION_SUMMARY = "generation_summary"
    OPTIMIZATION_ANALYSIS = "optimization_analysis"
    ROBUSTNESS_REPORT = "robustness_report"
    PERFORMANCE_COMPARISON = "performance_comparison"
    INDICATOR_ANALYSIS = "indicator_analysis"
    STRATEGY_BREAKDOWN = "strategy_breakdown"
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    EXPORT_SUMMARY = "export_summary"

class LogLevel(Enum):
    """Log levels for EA generation events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    METRIC = "METRIC"  # Special level for performance metrics

@dataclass
class EAGenerationEvent:
    """Individual event in EA generation process"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    level: LogLevel = LogLevel.INFO
    ea_id: Optional[str] = None
    template_name: Optional[str] = None
    phase: str = ""  # generation, optimization, robustness, export
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True
    error_details: Optional[str] = None

@dataclass
class GenerationMetrics:
    """Metrics for tracking EA generation performance"""
    total_eas_generated: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    average_generation_time: float = 0.0
    total_optimization_runs: int = 0
    successful_optimizations: int = 0
    average_optimization_time: float = 0.0
    total_robustness_tests: int = 0
    robustness_pass_rate: float = 0.0
    indicators_used: Dict[str, int] = field(default_factory=dict)
    validation_statistics: Dict[str, float] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType
    output_format: str = "html"  # html, pdf, json, csv
    include_charts: bool = True
    include_detailed_logs: bool = False
    time_range_hours: int = 24
    max_entries: int = 1000
    template_path: Optional[str] = None
    output_path: Optional[str] = None

class QNTIEAReportingSystem:
    """Comprehensive reporting system for EA generation"""
    
    def __init__(self, db_path: str = "qnti_ea_reports.db", log_level: LogLevel = LogLevel.INFO):
        self.db_path = Path(db_path)
        self.log_level = log_level
        self.logger = self._setup_logger()
        
        # Thread-safe event queue
        self.event_queue = deque(maxlen=10000)
        self.metrics = GenerationMetrics()
        self.lock = threading.Lock()
        
        # Database setup
        self._init_database()
        
        # Report templates
        if JINJA2_AVAILABLE:
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader('qnti_reports/templates'),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
        else:
            self.template_env = None
        
        # Real-time tracking
        self.active_generations = {}
        self.performance_history = defaultdict(list)
        
        # Output directories
        self.reports_dir = Path("qnti_reports")
        self.charts_dir = self.reports_dir / "charts"
        self.exports_dir = self.reports_dir / "exports"
        
        for dir_path in [self.reports_dir, self.charts_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger.info("QNTI EA Reporting System initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger("QNTI_EA_Reporting")
        logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler("qnti_ea_generation.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    ea_id TEXT,
                    template_name TEXT,
                    phase TEXT,
                    message TEXT,
                    data TEXT,
                    duration_ms REAL,
                    success BOOLEAN,
                    error_details TEXT
                )
            """)
            
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    ea_id TEXT,
                    additional_data TEXT
                )
            """)
            
            # EA Results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ea_results (
                    ea_id TEXT PRIMARY KEY,
                    template_name TEXT NOT NULL,
                    creation_time TEXT NOT NULL,
                    optimization_results TEXT,
                    robustness_results TEXT,
                    performance_metrics TEXT,
                    validation_status TEXT,
                    generation_time_ms REAL,
                    total_indicators INTEGER,
                    optimization_method TEXT
                )
            """)
            
            conn.commit()
    
    def log_event(self, event: EAGenerationEvent):
        """Log an EA generation event"""
        with self.lock:
            # Add to queue
            self.event_queue.append(event)
            
            # Log to file
            log_message = f"[{event.phase.upper()}] {event.message}"
            if event.ea_id:
                log_message = f"EA-{event.ea_id[:8]}: {log_message}"
            
            if event.level == LogLevel.DEBUG:
                self.logger.debug(log_message)
            elif event.level == LogLevel.INFO:
                self.logger.info(log_message)
            elif event.level == LogLevel.WARNING:
                self.logger.warning(log_message)
            elif event.level == LogLevel.ERROR:
                self.logger.error(log_message)
            elif event.level == LogLevel.CRITICAL:
                self.logger.critical(log_message)
            
            # Store in database
            self._store_event(event)
            
            # Update metrics
            self._update_metrics(event)
    
    def _store_event(self, event: EAGenerationEvent):
        """Store event in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO generation_events 
                    (event_id, timestamp, event_type, level, ea_id, template_name, 
                     phase, message, data, duration_ms, success, error_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.level.value,
                    event.ea_id,
                    event.template_name,
                    event.phase,
                    event.message,
                    json.dumps(event.data) if event.data else None,
                    event.duration_ms,
                    event.success,
                    event.error_details
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store event in database: {e}")
    
    def _update_metrics(self, event: EAGenerationEvent):
        """Update generation metrics based on event"""
        if event.phase == "generation":
            if event.event_type == "generation_started":
                self.active_generations[event.ea_id] = event.timestamp
            elif event.event_type == "generation_completed":
                self.metrics.total_eas_generated += 1
                if event.success:
                    self.metrics.successful_generations += 1
                else:
                    self.metrics.failed_generations += 1
                
                # Calculate generation time
                if event.ea_id in self.active_generations:
                    start_time = self.active_generations.pop(event.ea_id)
                    duration = (event.timestamp - start_time).total_seconds()
                    self._update_average_time("generation", duration)
        
        elif event.phase == "optimization":
            if event.event_type == "optimization_completed":
                self.metrics.total_optimization_runs += 1
                if event.success:
                    self.metrics.successful_optimizations += 1
                if event.duration_ms:
                    self._update_average_time("optimization", event.duration_ms / 1000)
        
        elif event.phase == "robustness":
            if event.event_type == "robustness_test_completed":
                self.metrics.total_robustness_tests += 1
        
        # Track indicator usage
        if "indicators" in event.data:
            for indicator in event.data["indicators"]:
                self.metrics.indicators_used[indicator] = \
                    self.metrics.indicators_used.get(indicator, 0) + 1
    
    def _update_average_time(self, phase: str, duration: float):
        """Update average time calculations"""
        if phase == "generation":
            current_avg = self.metrics.average_generation_time
            count = self.metrics.successful_generations
            self.metrics.average_generation_time = \
                ((current_avg * (count - 1)) + duration) / count
        elif phase == "optimization":
            current_avg = self.metrics.average_optimization_time
            count = self.metrics.successful_optimizations
            self.metrics.average_optimization_time = \
                ((current_avg * (count - 1)) + duration) / count
    
    def log_generation_start(self, ea_id: str, template_name: str, indicators: List[str]):
        """Log the start of EA generation"""
        event = EAGenerationEvent(
            event_type="generation_started",
            level=LogLevel.INFO,
            ea_id=ea_id,
            template_name=template_name,
            phase="generation",
            message=f"Starting EA generation for template '{template_name}'",
            data={"indicators": indicators, "indicator_count": len(indicators)}
        )
        self.log_event(event)
    
    def log_generation_complete(self, ea_id: str, success: bool, duration_ms: float, 
                              performance_metrics: Dict[str, float]):
        """Log completion of EA generation"""
        event = EAGenerationEvent(
            event_type="generation_completed",
            level=LogLevel.INFO if success else LogLevel.ERROR,
            ea_id=ea_id,
            phase="generation",
            message="EA generation completed successfully" if success else "EA generation failed",
            data={"performance_metrics": performance_metrics},
            duration_ms=duration_ms,
            success=success
        )
        self.log_event(event)
    
    def log_optimization_result(self, ea_id: str, method: str, result: Dict[str, Any], 
                              duration_ms: float):
        """Log optimization results"""
        event = EAGenerationEvent(
            event_type="optimization_completed",
            level=LogLevel.INFO,
            ea_id=ea_id,
            phase="optimization",
            message=f"Optimization completed using {method}",
            data={
                "method": method,
                "best_parameters": result.get("best_parameters", {}),
                "best_score": result.get("best_score", 0),
                "iterations": result.get("iterations", 0)
            },
            duration_ms=duration_ms,
            success=True
        )
        self.log_event(event)
    
    def log_robustness_test(self, ea_id: str, test_type: str, result: Dict[str, Any]):
        """Log robustness test results"""
        passed = result.get("passed", False)
        event = EAGenerationEvent(
            event_type="robustness_test_completed",
            level=LogLevel.INFO if passed else LogLevel.WARNING,
            ea_id=ea_id,
            phase="robustness",
            message=f"Robustness test '{test_type}' {'passed' if passed else 'failed'}",
            data=result,
            success=passed
        )
        self.log_event(event)
    
    def generate_report(self, config: ReportConfig) -> str:
        """Generate comprehensive report based on configuration"""
        self.logger.info(f"Generating {config.report_type.value} report")
        
        if config.report_type == ReportType.GENERATION_SUMMARY:
            return self._generate_generation_summary(config)
        elif config.report_type == ReportType.OPTIMIZATION_ANALYSIS:
            return self._generate_optimization_analysis(config)
        elif config.report_type == ReportType.ROBUSTNESS_REPORT:
            return self._generate_robustness_report(config)
        elif config.report_type == ReportType.PERFORMANCE_COMPARISON:
            return self._generate_performance_comparison(config)
        elif config.report_type == ReportType.INDICATOR_ANALYSIS:
            return self._generate_indicator_analysis(config)
        elif config.report_type == ReportType.REAL_TIME_DASHBOARD:
            return self._generate_realtime_dashboard(config)
        else:
            raise ValueError(f"Unsupported report type: {config.report_type}")
    
    def _generate_generation_summary(self, config: ReportConfig) -> str:
        """Generate EA generation summary report"""
        # Collect recent events
        cutoff_time = datetime.now() - timedelta(hours=config.time_range_hours)
        recent_events = [e for e in self.event_queue if e.timestamp >= cutoff_time]
        
        # Generate statistics
        stats = {
            "time_period": f"Last {config.time_range_hours} hours",
            "total_events": len(recent_events),
            "total_eas": self.metrics.total_eas_generated,
            "success_rate": (self.metrics.successful_generations / 
                           max(self.metrics.total_eas_generated, 1)) * 100,
            "average_generation_time": self.metrics.average_generation_time,
            "active_generations": len(self.active_generations),
            "most_used_indicators": dict(sorted(self.metrics.indicators_used.items(), 
                                              key=lambda x: x[1], reverse=True)[:10])
        }
        
        # Create report
        if config.output_format == "html":
            return self._create_html_report("generation_summary", stats, recent_events)
        elif config.output_format == "json":
            return json.dumps(stats, indent=2, default=str)
        else:
            return self._create_text_report(stats)
    
    def _generate_optimization_analysis(self, config: ReportConfig) -> str:
        """Generate optimization analysis report"""
        # Query optimization events from database
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM generation_events 
                WHERE phase = 'optimization' 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(config.time_range_hours), conn)
        
        # Analyze optimization performance
        analysis = {
            "total_optimizations": len(df),
            "success_rate": (df['success'].sum() / len(df)) * 100 if len(df) > 0 else 0,
            "average_duration": df['duration_ms'].mean() if len(df) > 0 else 0,
            "method_performance": {},
            "convergence_analysis": {}
        }
        
        # Method-specific analysis
        for _, row in df.iterrows():
            if row['data']:
                data = json.loads(row['data'])
                method = data.get('method', 'unknown')
                if method not in analysis["method_performance"]:
                    analysis["method_performance"][method] = {
                        "count": 0, "avg_score": 0, "avg_iterations": 0
                    }
                
                analysis["method_performance"][method]["count"] += 1
                if data.get('best_score'):
                    analysis["method_performance"][method]["avg_score"] += data['best_score']
                if data.get('iterations'):
                    analysis["method_performance"][method]["avg_iterations"] += data['iterations']
        
        # Calculate averages
        for method_data in analysis["method_performance"].values():
            if method_data["count"] > 0:
                method_data["avg_score"] /= method_data["count"]
                method_data["avg_iterations"] /= method_data["count"]
        
        return json.dumps(analysis, indent=2, default=str)
    
    def _generate_robustness_report(self, config: ReportConfig) -> str:
        """Generate robustness testing report"""
        # Query robustness events
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM generation_events 
                WHERE phase = 'robustness' 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(config.time_range_hours), conn)
        
        # Analyze robustness test results
        test_results = defaultdict(lambda: {"passed": 0, "total": 0})
        
        for _, row in df.iterrows():
            if row['data']:
                data = json.loads(row['data'])
                test_type = data.get('test_type', 'unknown')
                test_results[test_type]["total"] += 1
                if row['success']:
                    test_results[test_type]["passed"] += 1
        
        # Calculate pass rates
        analysis = {}
        for test_type, results in test_results.items():
            pass_rate = (results["passed"] / results["total"]) * 100 if results["total"] > 0 else 0
            analysis[test_type] = {
                "total_tests": results["total"],
                "passed": results["passed"],
                "pass_rate": pass_rate
            }
        
        return json.dumps(analysis, indent=2, default=str)
    
    def _generate_performance_comparison(self, config: ReportConfig) -> str:
        """Generate performance comparison report"""
        # Query EA results from database
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM ea_results 
                WHERE creation_time >= datetime('now', '-{} hours')
                ORDER BY creation_time DESC
            """.format(config.time_range_hours), conn)
        
        if len(df) == 0:
            return json.dumps({"message": "No EA results found in specified time range"})
        
        # Parse performance metrics
        performance_data = []
        for _, row in df.iterrows():
            if row['performance_metrics']:
                metrics = json.loads(row['performance_metrics'])
                metrics['ea_id'] = row['ea_id']
                metrics['template_name'] = row['template_name']
                metrics['validation_status'] = row['validation_status']
                performance_data.append(metrics)
        
        # Generate comparison analysis
        comparison = {
            "total_eas": len(performance_data),
            "top_performers": sorted(performance_data, 
                                   key=lambda x: x.get('profit_factor', 0), reverse=True)[:5],
            "average_metrics": {},
            "validation_summary": {}
        }
        
        # Calculate average metrics
        if performance_data:
            metric_keys = ['profit_factor', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            for key in metric_keys:
                values = [d.get(key, 0) for d in performance_data if d.get(key) is not None]
                comparison["average_metrics"][key] = np.mean(values) if values else 0
        
        # Validation status summary
        validation_counts = defaultdict(int)
        for data in performance_data:
            validation_counts[data.get('validation_status', 'unknown')] += 1
        comparison["validation_summary"] = dict(validation_counts)
        
        return json.dumps(comparison, indent=2, default=str)
    
    def _generate_indicator_analysis(self, config: ReportConfig) -> str:
        """Generate indicator usage analysis"""
        analysis = {
            "total_indicators_available": len(self.metrics.indicators_used),
            "most_popular": dict(sorted(self.metrics.indicators_used.items(), 
                                      key=lambda x: x[1], reverse=True)[:20]),
            "usage_distribution": self._calculate_indicator_distribution(),
            "correlation_analysis": self._analyze_indicator_performance()
        }
        
        return json.dumps(analysis, indent=2, default=str)
    
    def _calculate_indicator_distribution(self) -> Dict[str, Any]:
        """Calculate indicator usage distribution"""
        usage_counts = list(self.metrics.indicators_used.values())
        if not usage_counts:
            return {}
        
        return {
            "mean_usage": np.mean(usage_counts),
            "median_usage": np.median(usage_counts),
            "std_dev": np.std(usage_counts),
            "max_usage": max(usage_counts),
            "min_usage": min(usage_counts)
        }
    
    def _analyze_indicator_performance(self) -> Dict[str, Any]:
        """Analyze correlation between indicators and EA performance"""
        # This would require more sophisticated analysis
        # For now, return placeholder structure
        return {
            "high_performance_indicators": [],
            "low_performance_indicators": [],
            "correlation_matrix": {},
            "analysis_note": "Detailed correlation analysis requires more historical data"
        }
    
    def _generate_realtime_dashboard(self, config: ReportConfig) -> str:
        """Generate real-time dashboard data"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "active_generations": len(self.active_generations),
            "recent_completions": len([e for e in self.event_queue 
                                     if e.timestamp >= datetime.now() - timedelta(hours=1)]),
            "current_metrics": asdict(self.metrics),
            "system_health": {
                "queue_size": len(self.event_queue),
                "success_rate_24h": self._calculate_recent_success_rate(),
                "average_generation_time_1h": self._calculate_recent_avg_time()
            },
            "recent_events": [asdict(e) for e in list(self.event_queue)[-10:]]
        }
        
        return json.dumps(dashboard_data, indent=2, default=str)
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for last 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        recent_events = [e for e in self.event_queue 
                        if e.timestamp >= cutoff and e.event_type == "generation_completed"]
        
        if not recent_events:
            return 0.0
        
        successful = sum(1 for e in recent_events if e.success)
        return (successful / len(recent_events)) * 100
    
    def _calculate_recent_avg_time(self) -> float:
        """Calculate average generation time for last hour"""
        cutoff = datetime.now() - timedelta(hours=1)
        recent_events = [e for e in self.event_queue 
                        if e.timestamp >= cutoff and e.duration_ms is not None]
        
        if not recent_events:
            return 0.0
        
        return np.mean([e.duration_ms for e in recent_events])
    
    def _create_html_report(self, template_name: str, stats: Dict[str, Any], 
                          events: List[EAGenerationEvent]) -> str:
        """Create HTML report using Jinja2 template"""
        try:
            template = self.template_env.get_template(f"{template_name}.html")
            return template.render(stats=stats, events=events, 
                                 generation_time=datetime.now())
        except Exception as e:
            self.logger.warning(f"Failed to load template {template_name}: {e}")
            return self._create_text_report(stats)
    
    def _create_text_report(self, stats: Dict[str, Any]) -> str:
        """Create simple text report as fallback"""
        report = "QNTI EA Generation Report\n"
        report += "=" * 50 + "\n\n"
        
        for key, value in stats.items():
            report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        return report
    
    def export_data(self, format_type: str = "csv", time_range_hours: int = 24) -> str:
        """Export generation data in various formats"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Export events
            events_df = pd.read_sql_query("""
                SELECT * FROM generation_events 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, conn, params=[cutoff_time.isoformat()])
            
            # Export metrics
            metrics_df = pd.read_sql_query("""
                SELECT * FROM generation_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, conn, params=[cutoff_time.isoformat()])
            
            # Export EA results
            results_df = pd.read_sql_query("""
                SELECT * FROM ea_results 
                WHERE creation_time >= ?
                ORDER BY creation_time DESC
            """, conn, params=[cutoff_time.isoformat()])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "csv":
            events_path = self.exports_dir / f"events_{timestamp}.csv"
            metrics_path = self.exports_dir / f"metrics_{timestamp}.csv"
            results_path = self.exports_dir / f"results_{timestamp}.csv"
            
            events_df.to_csv(events_path, index=False)
            metrics_df.to_csv(metrics_path, index=False)
            results_df.to_csv(results_path, index=False)
            
            return f"Data exported to: {events_path}, {metrics_path}, {results_path}"
        
        elif format_type == "json":
            export_data = {
                "events": events_df.to_dict('records'),
                "metrics": metrics_df.to_dict('records'),
                "results": results_df.to_dict('records'),
                "export_timestamp": datetime.now().isoformat()
            }
            
            export_path = self.exports_dir / f"ea_data_{timestamp}.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return f"Data exported to: {export_path}"
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics for dashboard"""
        return {
            "active_generations": len(self.active_generations),
            "total_generated": self.metrics.total_eas_generated,
            "success_rate": (self.metrics.successful_generations / 
                           max(self.metrics.total_eas_generated, 1)) * 100,
            "queue_size": len(self.event_queue),
            "recent_activity": len([e for e in self.event_queue 
                                  if e.timestamp >= datetime.now() - timedelta(minutes=5)])
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database and logs"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean events
            events_deleted = conn.execute("""
                DELETE FROM generation_events 
                WHERE timestamp < ?
            """, [cutoff_date.isoformat()]).rowcount
            
            # Clean metrics
            metrics_deleted = conn.execute("""
                DELETE FROM generation_metrics 
                WHERE timestamp < ?
            """, [cutoff_date.isoformat()]).rowcount
            
            conn.commit()
        
        self.logger.info(f"Cleaned up {events_deleted} events and {metrics_deleted} metrics older than {days_to_keep} days")
    
    def shutdown(self):
        """Shutdown reporting system gracefully"""
        self.logger.info("Shutting down QNTI EA Reporting System")
        
        # Final metrics update
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO generation_metrics 
                (timestamp, metric_name, metric_value, additional_data)
                VALUES (?, 'final_session_stats', ?, ?)
            """, [
                datetime.now().isoformat(),
                self.metrics.total_eas_generated,
                json.dumps(asdict(self.metrics))
            ])
            conn.commit()

def main():
    """Test the reporting system"""
    reporting = QNTIEAReportingSystem()
    
    # Simulate some events
    reporting.log_generation_start("test-ea-1", "RSI_MA_Strategy", ["RSI", "SMA", "EMA"])
    time.sleep(1)
    reporting.log_generation_complete("test-ea-1", True, 5000, {"profit_factor": 1.5})
    
    # Generate test report
    config = ReportConfig(
        report_type=ReportType.GENERATION_SUMMARY,
        output_format="json",
        time_range_hours=1
    )
    
    report = reporting.generate_report(config)
    print("Generated Report:")
    print(report)
    
    reporting.shutdown()

if __name__ == "__main__":
    main() 