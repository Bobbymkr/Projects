"""
Elite Performance Testing Framework for Adaptive Traffic Control System

This module implements world-class performance testing capabilities including:
- Load testing with realistic traffic patterns
- Stress testing to identify breaking points
- Scalability testing for horizontal and vertical scaling
- Real-time performance monitoring and analysis
"""

import os
import sys
import time
import threading
import multiprocessing
import statistics
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logger = logging.getLogger(__name__)

class PerformanceTestType(Enum):
    """Performance test types."""
    LOAD = "load_test"
    STRESS = "stress_test"
    SPIKE = "spike_test"
    ENDURANCE = "endurance_test"
    SCALABILITY = "scalability_test"

class PerformanceMetric(Enum):
    """Performance metrics to track."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"

@dataclass
class PerformanceResult:
    """Performance test result data."""
    metric: PerformanceMetric
    timestamp: datetime
    value: float
    unit: str
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadTestConfig:
    """Load test configuration."""
    virtual_users: int
    duration_seconds: int
    ramp_up_seconds: int
    think_time_seconds: float = 1.0

@dataclass
class PerformanceTestReport:
    """Comprehensive performance test report."""
    test_id: str
    test_type: PerformanceTestType
    start_time: datetime
    end_time: datetime
    configuration: Dict[str, Any]
    results: List[PerformanceResult]
    summary_statistics: Dict[str, Dict[str, float]]
    recommendations: List[str]
    sla_compliance: Dict[str, bool]

class SystemMonitor:
    """Real-time system performance monitor."""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.monitoring = False
        self.results: List[PerformanceResult] = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = datetime.now()
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.results.append(PerformanceResult(
                metric=PerformanceMetric.CPU_USAGE,
                timestamp=timestamp,
                value=cpu_percent,
                unit="percent"
            ))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            self.results.append(PerformanceResult(
                metric=PerformanceMetric.MEMORY_USAGE,
                timestamp=timestamp,
                value=memory.percent,
                unit="percent"
            ))
            
            time.sleep(self.interval)
    
    def get_results(self) -> List[PerformanceResult]:
        """Get monitoring results."""
        return self.results.copy()

class LoadTester:
    """Advanced load testing framework."""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.results: List[PerformanceResult] = []
        
    def run_load_test(self, config: LoadTestConfig, target_function: Callable, 
                     target_args: Tuple = (), target_kwargs: Dict = None) -> PerformanceTestReport:
        """Run comprehensive load test."""
        if target_kwargs is None:
            target_kwargs = {}
            
        logger.info(f"Starting load test with {config.virtual_users} virtual users")
        
        test_id = self._generate_test_id("LOAD")
        start_time = datetime.now()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Execute load test
            load_results = self._execute_load_test(config, target_function, target_args, target_kwargs)
            self.results.extend(load_results)
            
        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            
        end_time = datetime.now()
        
        # Collect all results
        all_results = self.results + self.system_monitor.get_results()
        
        # Generate report
        report = self._generate_performance_report(
            test_id=test_id,
            test_type=PerformanceTestType.LOAD,
            start_time=start_time,
            end_time=end_time,
            configuration=config.__dict__,
            results=all_results
        )
        
        logger.info(f"Load test completed: {test_id}")
        return report
    
    def _execute_load_test(self, config: LoadTestConfig, target_function: Callable,
                          target_args: Tuple, target_kwargs: Dict) -> List[PerformanceResult]:
        """Execute the actual load test."""
        results = []
        
        # Use ThreadPoolExecutor for concurrent load
        with ThreadPoolExecutor(max_workers=min(config.virtual_users, 100)) as executor:
            futures = []
            
            # Submit load test tasks
            for user_id in range(config.virtual_users):
                future = executor.submit(
                    self._virtual_user_simulation,
                    user_id=user_id,
                    duration=config.duration_seconds,
                    think_time=config.think_time_seconds,
                    target_function=target_function,
                    target_args=target_args,
                    target_kwargs=target_kwargs
                )
                futures.append(future)
            
            # Collect results from all virtual users
            for future in as_completed(futures):
                try:
                    user_results = future.result(timeout=config.duration_seconds + 60)
                    results.extend(user_results)
                except Exception as e:
                    logger.error(f"Virtual user execution failed: {str(e)}")
        
        return results
    
    def _virtual_user_simulation(self, user_id: int, duration: int, think_time: float, 
                                target_function: Callable, target_args: Tuple, 
                                target_kwargs: Dict) -> List[PerformanceResult]:
        """Simulate a single virtual user."""
        results = []
        start_time = time.time()
        end_time = start_time + duration
        request_count = 0
        error_count = 0
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                # Execute target function
                result = target_function(*target_args, **target_kwargs)
                
                # Record successful response time
                response_time = (time.time() - request_start) * 1000  # Convert to ms
                results.append(PerformanceResult(
                    metric=PerformanceMetric.RESPONSE_TIME,
                    timestamp=datetime.now(),
                    value=response_time,
                    unit="milliseconds",
                    additional_data={"user_id": user_id, "success": True}
                ))
                
            except Exception as e:
                # Record error
                error_count += 1
                results.append(PerformanceResult(
                    metric=PerformanceMetric.ERROR_RATE,
                    timestamp=datetime.now(),
                    value=1.0,
                    unit="count",
                    additional_data={"user_id": user_id, "error": str(e)}
                ))
            
            request_count += 1
            
            # Think time between requests
            if think_time > 0:
                time.sleep(think_time)
        
        # Record throughput for this user
        total_time = time.time() - start_time
        throughput = request_count / total_time if total_time > 0 else 0
        results.append(PerformanceResult(
            metric=PerformanceMetric.THROUGHPUT,
            timestamp=datetime.now(),
            value=throughput,
            unit="requests_per_second",
            additional_data={"user_id": user_id, "total_requests": request_count}
        ))
        
        return results
    
    def _generate_test_id(self, test_type: str) -> str:
        """Generate unique test identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{test_type}_{timestamp}_{os.getpid()}"
    
    def _generate_performance_report(self, test_id: str, test_type: PerformanceTestType,
                                   start_time: datetime, end_time: datetime,
                                   configuration: Dict[str, Any],
                                   results: List[PerformanceResult]) -> PerformanceTestReport:
        """Generate comprehensive performance report."""
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary_stats)
        
        # Check SLA compliance
        sla_compliance = self._check_sla_compliance(summary_stats)
        
        return PerformanceTestReport(
            test_id=test_id,
            test_type=test_type,
            start_time=start_time,
            end_time=end_time,
            configuration=configuration,
            results=results,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            sla_compliance=sla_compliance
        )
    
    def _calculate_summary_statistics(self, results: List[PerformanceResult]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for performance metrics."""
        stats = {}
        
        # Group results by metric
        metric_data = {}
        for result in results:
            if result.metric not in metric_data:
                metric_data[result.metric] = []
            metric_data[result.metric].append(result.value)
        
        # Calculate statistics for each metric
        for metric, values in metric_data.items():
            if values:
                stats[metric.value] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return stats
    
    def _generate_recommendations(self, summary_stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze response times
        if 'response_time' in summary_stats:
            rt_stats = summary_stats['response_time']
            if rt_stats['p95'] > 1000:  # 1 second
                recommendations.append("Optimize response times - P95 exceeds 1 second")
            elif rt_stats['p95'] > 500:  # 500ms
                recommendations.append("Consider response time optimization - P95 approaching limits")
        
        # Analyze resource utilization
        if 'cpu_usage' in summary_stats:
            cpu_stats = summary_stats['cpu_usage']
            if cpu_stats['mean'] > 80:
                recommendations.append("High CPU utilization detected - consider optimization")
        
        if 'memory_usage' in summary_stats:
            mem_stats = summary_stats['memory_usage']
            if mem_stats['mean'] > 85:
                recommendations.append("High memory utilization detected - review memory usage")
        
        # Add general recommendations
        recommendations.extend([
            "Implement continuous performance monitoring",
            "Establish performance baselines and alerts",
            "Regular performance regression testing"
        ])
        
        return recommendations
    
    def _check_sla_compliance(self, summary_stats: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check SLA compliance against defined thresholds."""
        sla_compliance = {}
        
        # Check response time SLA
        if 'response_time' in summary_stats:
            rt_stats = summary_stats['response_time']
            sla_compliance['response_time_p95'] = rt_stats['p95'] <= 500  # 500ms SLA
            sla_compliance['response_time_p99'] = rt_stats['p99'] <= 1000  # 1 second SLA
        
        # Check error rate SLA
        if 'error_rate' in summary_stats:
            error_stats = summary_stats['error_rate']
            total_requests = sum(len(results) for results in [summary_stats.get('response_time', {}).get('count', [1])])
            error_rate = (error_stats['count'] / total_requests * 100) if total_requests > 0 else 0
            sla_compliance['error_rate'] = error_rate <= 1  # 1% error rate SLA
        
        return sla_compliance

class StressTester:
    """Advanced stress testing framework."""
    
    def __init__(self):
        self.load_tester = LoadTester()
    
    def run_stress_test(self, target_function: Callable, initial_load: int = 10,
                       max_load: int = 500, increment: int = 50,
                       duration_per_step: int = 60) -> PerformanceTestReport:
        """Run comprehensive stress test to find breaking point."""
        logger.info(f"Starting stress test from {initial_load} to {max_load} users")
        
        test_id = f"STRESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        all_results = []
        
        current_load = initial_load
        breaking_point = None
        
        while current_load <= max_load:
            logger.info(f"Testing with {current_load} concurrent users")
            
            # Configure load test for current stress level
            config = LoadTestConfig(
                virtual_users=current_load,
                duration_seconds=duration_per_step,
                ramp_up_seconds=10,
                think_time_seconds=0.1
            )
            
            # Run load test at current stress level
            step_report = self.load_tester.run_load_test(config, target_function)
            all_results.extend(step_report.results)
            
            # Check if system is breaking
            if self._is_system_breaking(step_report):
                breaking_point = current_load
                logger.warning(f"Breaking point detected at {current_load} users")
                break
            
            current_load += increment
        
        end_time = datetime.now()
        
        # Generate stress test report with breaking point analysis
        summary_stats = self.load_tester._calculate_summary_statistics(all_results)
        
        return PerformanceTestReport(
            test_id=test_id,
            test_type=PerformanceTestType.STRESS,
            start_time=start_time,
            end_time=end_time,
            configuration={
                'breaking_point': breaking_point,
                'max_tested_load': current_load,
                'test_type': 'stress_test'
            },
            results=all_results,
            summary_statistics=summary_stats,
            recommendations=self._generate_stress_recommendations(breaking_point, current_load),
            sla_compliance={}
        )
    
    def _is_system_breaking(self, step_report: PerformanceTestReport) -> bool:
        """Determine if system is at breaking point."""
        # Check error rate
        if 'error_rate' in step_report.summary_statistics:
            error_count = step_report.summary_statistics['error_rate'].get('count', 0)
            if error_count > 10:  # High number of errors indicates breaking
                return True
        
        # Check response time degradation
        if 'response_time' in step_report.summary_statistics:
            rt_stats = step_report.summary_statistics['response_time']
            if rt_stats['p95'] > 5000:  # 5 seconds indicates severe degradation
                return True
        
        return False
    
    def _generate_stress_recommendations(self, breaking_point: Optional[int], 
                                      max_tested_load: int) -> List[str]:
        """Generate stress test specific recommendations."""
        recommendations = []
        
        if breaking_point:
            recommendations.extend([
                f"System breaking point identified at {breaking_point} concurrent users",
                f"Recommended safe operating load: {breaking_point * 0.7:.0f} users",
                "Implement load balancing to handle higher loads",
                "Consider horizontal scaling for capacity expansion"
            ])
        else:
            recommendations.extend([
                f"System handled {max_tested_load} users without breaking",
                "Continue stress testing with higher loads if needed",
                "System shows good resilience under stress"
            ])
        
        return recommendations

class ScalabilityValidator:
    """Scalability testing and validation framework."""
    
    def __init__(self):
        self.load_tester = LoadTester()
    
    def test_horizontal_scalability(self, target_function: Callable,
                                  instance_counts: List[int] = [1, 2, 4],
                                  load_per_instance: int = 100) -> Dict[str, PerformanceTestReport]:
        """Test horizontal scalability across multiple instances."""
        logger.info("Starting horizontal scalability test")
        
        results = {}
        
        for instance_count in instance_counts:
            logger.info(f"Testing with {instance_count} instances")
            
            total_load = instance_count * load_per_instance
            config = LoadTestConfig(
                virtual_users=total_load,
                duration_seconds=180,  # 3 minutes
                ramp_up_seconds=30,
                think_time_seconds=0.5
            )
            
            # Run load test for current instance configuration
            report = self.load_tester.run_load_test(config, target_function)
            results[f"{instance_count}_instances"] = report
        
        return results
    
    def analyze_scalability_efficiency(self, results: Dict[str, PerformanceTestReport]) -> Dict[str, Any]:
        """Analyze scalability efficiency from test results."""
        analysis = {
            'linear_scalability': False,
            'efficiency_score': 0.0,
            'optimal_configuration': None
        }
        
        # Extract throughput data for analysis
        throughputs = []
        configurations = []
        
        for config_name, report in results.items():
            if 'throughput' in report.summary_statistics:
                throughput = report.summary_statistics['throughput']['mean']
                throughputs.append(throughput)
                configurations.append(config_name)
        
        # Analyze scaling efficiency
        if len(throughputs) >= 2:
            base_throughput = throughputs[0]
            scaling_factors = []
            
            for i, throughput in enumerate(throughputs[1:], 1):
                expected_scaling = i + 1  # Expected linear scaling factor
                actual_scaling = throughput / base_throughput
                efficiency = actual_scaling / expected_scaling
                scaling_factors.append(efficiency)
            
            analysis['efficiency_score'] = statistics.mean(scaling_factors) if scaling_factors else 0.0
            analysis['linear_scalability'] = analysis['efficiency_score'] > 0.8
            
            # Find optimal configuration
            if scaling_factors:
                best_efficiency_idx = scaling_factors.index(max(scaling_factors))
                analysis['optimal_configuration'] = configurations[best_efficiency_idx + 1]
        
        return analysis

# Demo function for testing
def demo_traffic_function(*args, **kwargs):
    """Demo function simulating traffic environment operations."""
    import random
    import time
    
    # Simulate varying response times
    response_time = random.uniform(0.01, 0.05)  # 10-50ms
    time.sleep(response_time)
    
    # Simulate occasional errors under high load
    load_factor = kwargs.get('load_factor', 1.0)
    if load_factor > 3.0 and random.random() < 0.02:
        raise Exception("Simulated overload error")
    
    return {"status": "success", "response_time": response_time}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("ELITE PERFORMANCE TESTING FRAMEWORK")
    print("="*60)
    
    # Demo load testing
    load_tester = LoadTester()
    config = LoadTestConfig(virtual_users=20, duration_seconds=30, ramp_up_seconds=5)
    
    report = load_tester.run_load_test(config, demo_traffic_function)
    print(f"\nLoad Test Results:")
    print(f"Test ID: {report.test_id}")
    print(f"Duration: {report.end_time - report.start_time}")
    print(f"SLA Compliance: {report.sla_compliance}")
    print(f"Recommendations: {len(report.recommendations)} items")