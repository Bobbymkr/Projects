"""
Comprehensive Benchmarking Suite.

Industry-standard benchmarking framework for algorithm evaluation
and comparison.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    algorithm_name: str
    benchmark_name: str
    metrics: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkScenario:
    """Traffic scenario for benchmarking."""
    name: str
    description: str
    config: Dict[str, Any]
    expected_metrics: Dict[str, float]  # Baseline metrics
    difficulty: str  # 'easy', 'medium', 'hard'


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite.
    
    Supports:
    - Multiple benchmark scenarios
    - Algorithm comparison
    - Statistical analysis
    - Reproducibility
    """
    
    def __init__(self, name: str = "adaptive-traffic-benchmark"):
        """Initialize benchmark suite."""
        self.name = name
        self.scenarios: Dict[str, BenchmarkScenario] = {}
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, Dict[str, float]] = {}
    
    def add_scenario(self, scenario: BenchmarkScenario) -> None:
        """Add a benchmark scenario."""
        self.scenarios[scenario.name] = scenario
        self.baselines[scenario.name] = scenario.expected_metrics
        logger.info(f"Added benchmark scenario: {scenario.name}")
    
    def run_benchmark(
        self,
        algorithm: Any,
        scenario_name: str,
        num_runs: int = 5,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run benchmark for an algorithm on a scenario.
        
        Args:
            algorithm: Algorithm to benchmark
            scenario_name: Name of benchmark scenario
            num_runs: Number of runs for statistical significance
            **kwargs: Additional benchmark parameters
            
        Returns:
            Benchmark result
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        logger.info(f"Running benchmark: {algorithm.__class__.__name__} on {scenario_name}")
        
        # Run multiple times for statistical significance
        run_metrics = []
        execution_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Run algorithm on scenario
            metrics = self._run_scenario(algorithm, scenario, **kwargs)
            
            execution_time = time.time() - start_time
            
            run_metrics.append(metrics)
            execution_times.append(execution_time)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(run_metrics)
        aggregated_metrics["execution_time_mean"] = np.mean(execution_times)
        aggregated_metrics["execution_time_std"] = np.std(execution_times)
        
        # Compare with baseline
        baseline_metrics = self.baselines[scenario_name]
        improvement = self._calculate_improvement(aggregated_metrics, baseline_metrics)
        aggregated_metrics["improvement"] = improvement
        
        result = BenchmarkResult(
            algorithm_name=algorithm.__class__.__name__,
            benchmark_name=scenario_name,
            metrics=aggregated_metrics,
            execution_time=np.mean(execution_times),
            metadata={
                "num_runs": num_runs,
                "scenario_difficulty": scenario.difficulty,
                **kwargs,
            },
        )
        
        self.results.append(result)
        return result
    
    def _run_scenario(
        self,
        algorithm: Any,
        scenario: BenchmarkScenario,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run algorithm on a scenario (placeholder)."""
        # In production, this would:
        # 1. Create environment from scenario config
        # 2. Run algorithm
        # 3. Collect metrics
        
        # Placeholder metrics
        return {
            "wait_time_mean": np.random.uniform(1.0, 5.0),
            "wait_time_std": np.random.uniform(0.1, 0.5),
            "queue_length_mean": np.random.uniform(2.0, 8.0),
            "throughput": np.random.uniform(0.8, 1.2),
            "efficiency": np.random.uniform(0.7, 0.95),
        }
    
    def _aggregate_metrics(self, run_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple runs."""
        aggregated = {}
        
        # Get all metric names
        all_keys = set()
        for metrics in run_metrics:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [m.get(key, 0) for m in run_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
        
        return aggregated
    
    def _calculate_improvement(
        self,
        metrics: Dict[str, float],
        baseline: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate improvement over baseline."""
        improvement = {}
        
        for key, baseline_value in baseline.items():
            if key.endswith("_mean"):
                metric_key = key
            else:
                metric_key = f"{key}_mean"
            
            if metric_key in metrics:
                metric_value = metrics[metric_key]
                # Improvement percentage (lower is better for wait_time, queue_length)
                if "wait_time" in key or "queue" in key:
                    improvement[key] = (baseline_value - metric_value) / baseline_value * 100
                else:
                    improvement[key] = (metric_value - baseline_value) / baseline_value * 100
        
        return improvement
    
    def compare_algorithms(
        self,
        algorithms: List[Any],
        scenario_name: str,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms on a scenario.
        
        Args:
            algorithms: List of algorithms to compare
            scenario_name: Scenario to run
            num_runs: Number of runs per algorithm
            
        Returns:
            Comparison results
        """
        comparison = {
            "scenario": scenario_name,
            "algorithms": {},
            "best_algorithm": None,
            "comparison_metrics": {},
        }
        
        results = []
        for algorithm in algorithms:
            result = self.run_benchmark(algorithm, scenario_name, num_runs)
            comparison["algorithms"][result.algorithm_name] = {
                "metrics": result.metrics,
                "execution_time": result.execution_time,
            }
            results.append(result)
        
        # Find best algorithm
        if results:
            # Compare based on wait_time_mean (lower is better)
            best = min(
                results,
                key=lambda r: r.metrics.get("wait_time_mean", float('inf'))
            )
            comparison["best_algorithm"] = best.algorithm_name
        
        return comparison
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_suite": self.name,
            "timestamp": datetime.now().isoformat(),
            "scenarios": list(self.scenarios.keys()),
            "results": [
                {
                    "algorithm": r.algorithm_name,
                    "benchmark": r.benchmark_name,
                    "metrics": r.metrics,
                    "execution_time": r.execution_time,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "summary": self._generate_summary(),
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report saved to {output_path}")
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        summary = {
            "total_runs": len(self.results),
            "algorithms_tested": len(set(r.algorithm_name for r in self.results)),
            "scenarios_tested": len(set(r.benchmark_name for r in self.results)),
        }
        
        return summary


# Standard benchmark scenarios
def create_standard_scenarios() -> Dict[str, BenchmarkScenario]:
    """Create standard benchmark scenarios."""
    scenarios = {}
    
    # Easy scenario: Low traffic
    scenarios["low_traffic"] = BenchmarkScenario(
        name="low_traffic",
        description="Low traffic density scenario",
        config={
            "traffic_density": 0.3,
            "num_intersections": 1,
        },
        expected_metrics={
            "wait_time_mean": 3.5,
            "queue_length_mean": 4.0,
        },
        difficulty="easy",
    )
    
    # Medium scenario: Moderate traffic
    scenarios["moderate_traffic"] = BenchmarkScenario(
        name="moderate_traffic",
        description="Moderate traffic density scenario",
        config={
            "traffic_density": 0.6,
            "num_intersections": 1,
        },
        expected_metrics={
            "wait_time_mean": 5.5,
            "queue_length_mean": 7.0,
        },
        difficulty="medium",
    )
    
    # Hard scenario: High traffic
    scenarios["high_traffic"] = BenchmarkScenario(
        name="high_traffic",
        description="High traffic density scenario",
        config={
            "traffic_density": 0.9,
            "num_intersections": 1,
        },
        expected_metrics={
            "wait_time_mean": 8.5,
            "queue_length_mean": 12.0,
        },
        difficulty="hard",
    )
    
    return scenarios

