"""
Public Benchmarking Suite.

Industry-standard benchmarks and evaluation framework for traffic control algorithms.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BenchmarkDataset:
    """Standard benchmark dataset for traffic control."""
    
    def __init__(self, name: str, description: str):
        """Initialize benchmark dataset."""
        self.name = name
        self.description = description
        self.scenarios: List[Dict[str, Any]] = []
    
    def add_scenario(
        self,
        scenario_id: str,
        initial_state: Dict[str, Any],
        traffic_pattern: str,
        duration: int,
        expected_metrics: Optional[Dict[str, float]] = None
    ):
        """Add a benchmark scenario."""
        self.scenarios.append({
            "id": scenario_id,
            "initial_state": initial_state,
            "traffic_pattern": traffic_pattern,
            "duration": duration,
            "expected_metrics": expected_metrics or {},
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "scenarios": self.scenarios,
        }


class BenchmarkEvaluator:
    """Evaluate algorithms on benchmark datasets."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results: List[Dict[str, Any]] = []
    
    def evaluate_algorithm(
        self,
        algorithm_name: str,
        algorithm_func,
        dataset: BenchmarkDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an algorithm on a benchmark dataset.
        
        Args:
            algorithm_name: Name of the algorithm
            algorithm_func: Function that takes state and returns decision
            dataset: Benchmark dataset
            config: Algorithm configuration
            
        Returns:
            Evaluation results
        """
        config = config or {}
        results = {
            "algorithm": algorithm_name,
            "dataset": dataset.name,
            "timestamp": datetime.now().isoformat(),
            "scenarios": [],
            "overall_metrics": {},
        }
        
        total_wait_time = 0
        total_queue_length = 0
        total_decisions = 0
        decision_times = []
        
        for scenario in dataset.scenarios:
            scenario_results = self._evaluate_scenario(
                algorithm_func,
                scenario,
                config
            )
            results["scenarios"].append(scenario_results)
            
            # Aggregate metrics
            total_wait_time += scenario_results.get("avg_wait_time", 0)
            total_queue_length += scenario_results.get("avg_queue_length", 0)
            total_decisions += scenario_results.get("decision_count", 0)
            decision_times.extend(scenario_results.get("decision_times", []))
        
        # Calculate overall metrics
        num_scenarios = len(dataset.scenarios)
        results["overall_metrics"] = {
            "average_wait_time": total_wait_time / num_scenarios if num_scenarios > 0 else 0,
            "average_queue_length": total_queue_length / num_scenarios if num_scenarios > 0 else 0,
            "total_decisions": total_decisions,
            "average_decision_time_ms": np.mean(decision_times) if decision_times else 0,
            "p95_decision_time_ms": np.percentile(decision_times, 95) if decision_times else 0,
        }
        
        self.results.append(results)
        return results
    
    def _evaluate_scenario(
        self,
        algorithm_func,
        scenario: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate algorithm on a single scenario."""
        initial_state = scenario["initial_state"]
        wait_times = []
        queue_lengths = []
        decision_times = []
        
        # Simulate scenario
        current_state = initial_state.copy()
        for step in range(scenario.get("duration", 100)):
            # Measure decision time
            start_time = time.time()
            decision = algorithm_func(current_state, config)
            decision_time = (time.time() - start_time) * 1000  # Convert to ms
            decision_times.append(decision_time)
            
            # Simulate state evolution (simplified)
            # In real implementation, this would use SUMO or similar
            wait_times.append(np.mean(current_state.get("wait_times", [0])))
            queue_lengths.append(np.mean(current_state.get("queue_lengths", [0])))
            
            # Update state (simplified simulation)
            for i in range(len(current_state.get("queue_lengths", []))):
                if i == decision.get("phase", 0):
                    # Reduce queue for selected phase
                    current_state["queue_lengths"][i] = max(0, current_state["queue_lengths"][i] - 5)
                else:
                    # Increase queue for other phases
                    current_state["queue_lengths"][i] += 1
        
        return {
            "scenario_id": scenario["id"],
            "avg_wait_time": np.mean(wait_times) if wait_times else 0,
            "avg_queue_length": np.mean(queue_lengths) if queue_lengths else 0,
            "decision_count": len(decision_times),
            "decision_times": decision_times,
        }
    
    def compare_algorithms(
        self,
        algorithm_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple algorithm results."""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "algorithms": [],
            "rankings": {},
        }
        
        # Extract metrics for comparison
        algorithm_metrics = []
        for result in algorithm_results:
            metrics = result.get("overall_metrics", {})
            algorithm_metrics.append({
                "algorithm": result["algorithm"],
                "avg_wait_time": metrics.get("average_wait_time", 0),
                "avg_decision_time": metrics.get("average_decision_time_ms", 0),
            })
        
        # Rank by wait time (lower is better)
        sorted_by_wait = sorted(algorithm_metrics, key=lambda x: x["avg_wait_time"])
        comparison["rankings"]["by_wait_time"] = [
            {"algorithm": m["algorithm"], "wait_time": m["avg_wait_time"]}
            for m in sorted_by_wait
        ]
        
        # Rank by decision time (lower is better)
        sorted_by_decision = sorted(algorithm_metrics, key=lambda x: x["avg_decision_time"])
        comparison["rankings"]["by_decision_time"] = [
            {"algorithm": m["algorithm"], "decision_time_ms": m["avg_decision_time"]}
            for m in sorted_by_decision
        ]
        
        comparison["algorithms"] = algorithm_metrics
        
        return comparison
    
    def export_results(self, output_path: Path):
        """Export results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "results": self.results,
                "exported_at": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Results exported to {output_path}")


def create_standard_datasets() -> Dict[str, BenchmarkDataset]:
    """Create standard benchmark datasets."""
    datasets = {}
    
    # Light Traffic Dataset
    light_traffic = BenchmarkDataset(
        name="light_traffic",
        description="Light traffic conditions with low vehicle density"
    )
    light_traffic.add_scenario(
        scenario_id="light_1",
        initial_state={
            "queue_lengths": [2, 1, 3, 1],
            "wait_times": [5.0, 3.0, 7.0, 2.0],
            "arrival_rates": [0.1, 0.05, 0.15, 0.08],
        },
        traffic_pattern="light",
        duration=100,
    )
    datasets["light_traffic"] = light_traffic
    
    # Medium Traffic Dataset
    medium_traffic = BenchmarkDataset(
        name="medium_traffic",
        description="Medium traffic conditions with moderate vehicle density"
    )
    medium_traffic.add_scenario(
        scenario_id="medium_1",
        initial_state={
            "queue_lengths": [8, 6, 10, 5],
            "wait_times": [18.5, 14.2, 22.1, 12.3],
            "arrival_rates": [0.4, 0.3, 0.5, 0.25],
        },
        traffic_pattern="medium",
        duration=100,
    )
    datasets["medium_traffic"] = medium_traffic
    
    # Heavy Traffic Dataset
    heavy_traffic = BenchmarkDataset(
        name="heavy_traffic",
        description="Heavy traffic conditions with high vehicle density"
    )
    heavy_traffic.add_scenario(
        scenario_id="heavy_1",
        initial_state={
            "queue_lengths": [25, 30, 28, 22],
            "wait_times": [45.2, 52.1, 48.7, 38.9],
            "arrival_rates": [0.8, 0.9, 0.85, 0.75],
        },
        traffic_pattern="heavy",
        duration=100,
    )
    datasets["heavy_traffic"] = heavy_traffic
    
    return datasets


def main():
    """Main entry point for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run public benchmarks")
    parser.add_argument("--dataset", default="all", help="Dataset to use")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    
    args = parser.parse_args()
    
    # Create datasets
    datasets = create_standard_datasets()
    
    # Example: Evaluate algorithms
    evaluator = BenchmarkEvaluator()
    
    # This is a placeholder - in real implementation, algorithms would be evaluated
    print("Benchmarking suite ready.")
    print("Datasets available:", list(datasets.keys()))


if __name__ == "__main__":
    main()

