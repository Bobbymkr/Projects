#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for All Technologies.

This script benchmarks all 17+ technologies across 10 scenarios
as specified in the Perfect Score Execution Plan Week 1.

Usage:
    python scripts/benchmark_all_technologies.py --episodes 5000
    python scripts/benchmark_all_technologies.py --quick --technologies fuzzy_logic dqn
    python scripts/benchmark_all_technologies.py --scenarios rush_hour off_peak
"""

import json
import argparse
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scenarios.scenario_library import SCENARIOS, get_scenario
except ImportError:
    # Fallback if scenario library not found
    SCENARIOS = {}
    def get_scenario(name: str):
        return SCENARIOS.get(name)

# Import technologies (with graceful fallbacks)
try:
    from src.env.traffic_env import TrafficEnv
    from src.rl.dqn_agent import DQNAgent, DQNConfig
    from src.control.fuzzy_control import FuzzyController
    from src.control.webster_method import WebsterMethod
    from src.optimization.genetic_algo import GeneticAlgorithm
    from src.optimization.pso import ParticleSwarmOptimizer
    from src.forecast.gnn_forecast import GNNForecaster
    from src.forecast.traffic_forecast import TrafficForecaster
except ImportError as e:
    print(f"Warning: Some imports failed: {e}", file=sys.stderr)

# Try to import advanced RL agents
try:
    from src.research.novel_algorithms.model_based_rl import ModelBasedRLAgent
    from src.research.novel_algorithms.hierarchical_rl import HierarchicalRLAgent
    from src.research.novel_algorithms.transformer_control import TransformerAgent
    from src.research.novel_algorithms.imitation_learning import BehavioralCloningAgent
    from src.research.novel_algorithms.bayesian_methods import BayesianAgent
except ImportError:
    # These may not be available, will handle gracefully
    ModelBasedRLAgent = None
    HierarchicalRLAgent = None
    TransformerAgent = None
    BehavioralCloningAgent = None
    BayesianAgent = None

# Load benchmark config
BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmarking" / "benchmark_config.json"
if BENCHMARK_CONFIG.exists():
    with open(BENCHMARK_CONFIG) as f:
        config = json.load(f)
        EXPECTED_TECHNOLOGIES = config.get("technologies", [])
        DEFAULT_EPISODES = config.get("default_episodes", 5000)
        QUICK_EPISODES = config.get("quick_episodes", 100)
else:
    EXPECTED_TECHNOLOGIES = [
        "model_based_rl", "hierarchical_rl", "transformer_agent", "dqn",
        "fuzzy_logic", "webster", "genetic_algorithm", "pso",
        "gnn_forecast", "lstm_forecast", "imitation_learning",
        "bayesian_rl", "causal_rl"
    ]
    DEFAULT_EPISODES = 5000
    QUICK_EPISODES = 100


class BenchmarkMetrics:
    """Collect and aggregate benchmark metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.wait_times = []
        self.queue_lengths = []
        self.throughput = []
        self.inference_latencies = []
        self.cpu_usage = []
        self.memory_usage = []
        self.total_vehicles = 0
        self.total_wait_time = 0.0
        self.episodes_completed = 0
    
    def add_episode(self, episode_metrics: Dict[str, Any]):
        """Add metrics from a single episode."""
        self.wait_times.append(episode_metrics.get("avg_wait_time", 0))
        self.queue_lengths.append(episode_metrics.get("avg_queue_length", 0))
        self.throughput.append(episode_metrics.get("throughput", 0))
        self.inference_latencies.append(episode_metrics.get("inference_latency_ms", 0))
        self.cpu_usage.append(episode_metrics.get("cpu_usage_percent", 0))
        self.memory_usage.append(episode_metrics.get("memory_mb", 0))
        self.total_vehicles += episode_metrics.get("vehicles_processed", 0)
        self.total_wait_time += episode_metrics.get("total_wait_time", 0)
        self.episodes_completed += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.wait_times:
            return {}
        
        import numpy as np
        
        return {
            "avg_wait_time": np.mean(self.wait_times),
            "max_wait_time": np.max(self.wait_times),
            "95th_percentile_wait": np.percentile(self.wait_times, 95),
            "avg_queue_length": np.mean(self.queue_lengths),
            "throughput_vehicles_per_hour": np.mean(self.throughput),
            "avg_inference_latency_ms": np.mean(self.inference_latencies),
            "p95_inference_latency_ms": np.percentile(self.inference_latencies, 95) if self.inference_latencies else 0,
            "avg_cpu_usage_percent": np.mean(self.cpu_usage),
            "avg_memory_mb": np.mean(self.memory_usage),
            "episodes_completed": self.episodes_completed,
            "total_vehicles": self.total_vehicles
        }


def create_agent(technology: str, env_config: Dict[str, Any]) -> Any:
    """Create an agent instance for a given technology."""
    try:
        if technology == "dqn":
            obs_dim = env_config.get("num_lanes", 4) * 2  # queue + wait time
            action_dim = len(env_config.get("green_values", [10, 20, 30, 40]))
            return DQNAgent(obs_dim, action_dim, DQNConfig())
        
        elif technology == "fuzzy_logic":
            return FuzzyController()
        
        elif technology == "webster":
            return WebsterMethod()
        
        elif technology == "genetic_algorithm":
            return GeneticAlgorithm()
        
        elif technology == "pso":
            return ParticleSwarmOptimizer()
        
        elif technology == "gnn_forecast":
            num_nodes = env_config.get("num_lanes", 4)
            return GNNForecaster(num_nodes=num_nodes, input_dim=1, time_steps=1)
        
        elif technology == "lstm_forecast":
            return TrafficForecaster()
        
        elif technology == "model_based_rl" and ModelBasedRLAgent:
            return ModelBasedRLAgent()
        
        elif technology == "hierarchical_rl" and HierarchicalRLAgent:
            return HierarchicalRLAgent()
        
        elif technology == "transformer_agent" and TransformerAgent:
            return TransformerAgent()
        
        elif technology == "imitation_learning" and BehavioralCloningAgent:
            return BehavioralCloningAgent()
        
        elif technology == "bayesian_rl" and BayesianAgent:
            return BayesianAgent()
        
        else:
            print(f"Warning: Technology {technology} not implemented or not available", file=sys.stderr)
            return None
    
    except Exception as e:
        print(f"Error creating agent for {technology}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def run_episode(technology: str, scenario: str, env_config: Dict[str, Any], episode_num: int) -> Dict[str, Any]:
    """Run a single episode benchmark."""
    start_time = time.time()
    
    try:
        # Create environment
        env = TrafficEnv(env_config)
        obs, info = env.reset()
        
        # Create agent
        agent = create_agent(technology, env_config)
        if agent is None:
            return {"error": f"Could not create agent for {technology}"}
        
        # Run episode
        done = False
        total_reward = 0
        step_count = 0
        inference_times = []
        
        while not done:
            step_start = time.time()
            
            # Get action from agent
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs)
            elif hasattr(agent, 'compute_timing'):
                action = agent.compute_timing(obs)
            elif hasattr(agent, 'decide'):
                action = agent.decide(obs)
            elif hasattr(agent, 'get_action'):
                action = agent.get_action({'volumes': obs})
            else:
                # Default: random action
                action = env.action_space.sample()
            
            inference_time = (time.time() - step_start) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            step_count += 1
            
            # Safety limit
            if step_count > 10000:
                break
        
        # Calculate metrics
        total_wait_time = info.get("total_wait_time", 0)
        vehicles_processed = info.get("total_vehicles_processed", 1)
        avg_wait_time = total_wait_time / vehicles_processed if vehicles_processed > 0 else 0
        
        episode_time = time.time() - start_time
        
        return {
            "technology": technology,
            "scenario": scenario,
            "episode": episode_num,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": info.get("max_wait_time", 0),
            "avg_queue_length": info.get("avg_queue_length", 0),
            "throughput": vehicles_processed / (episode_time / 3600) if episode_time > 0 else 0,
            "inference_latency_ms": np.mean(inference_times) if inference_times else 0,
            "p95_inference_latency_ms": np.percentile(inference_times, 95) if inference_times else 0,
            "total_wait_time": total_wait_time,
            "vehicles_processed": vehicles_processed,
            "episode_time": episode_time,
            "total_reward": total_reward,
            "steps": step_count
        }
    
    except Exception as e:
        print(f"Error in episode {episode_num} for {technology}/{scenario}: {e}", file=sys.stderr)
        traceback.print_exc()
        return {
            "technology": technology,
            "scenario": scenario,
            "episode": episode_num,
            "error": str(e)
        }


def benchmark_technology_scenario(
    technology: str,
    scenario: str,
    episodes: int,
    env_config: Dict[str, Any],
    use_parallel: bool = False
) -> Dict[str, Any]:
    """Benchmark a technology on a specific scenario."""
    print(f"  Benchmarking {technology} on {scenario} ({episodes} episodes)...")
    
    metrics = BenchmarkMetrics()
    errors = []
    
    if use_parallel and episodes > 10:
        # Use multiprocessing for large episode counts
        with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            results = pool.starmap(
                run_episode,
                [(technology, scenario, env_config, i) for i in range(episodes)]
            )
    else:
        # Sequential execution
        results = [run_episode(technology, scenario, env_config, i) for i in range(episodes)]
    
    # Aggregate results
    for result in results:
        if "error" in result:
            errors.append(result)
        else:
            metrics.add_episode(result)
    
    summary = metrics.get_summary()
    summary["technology"] = technology
    summary["scenario"] = scenario
    summary["episodes"] = episodes
    summary["errors"] = len(errors)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark all technologies")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes per technology×scenario")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (100 episodes)")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios (default: all)")
    parser.add_argument("--technologies", nargs="+", help="Specific technologies (default: all)")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    
    args = parser.parse_args()
    
    # Determine episodes
    episodes = QUICK_EPISODES if args.quick else args.episodes
    
    # Get scenarios
    if args.scenarios:
        scenarios = [s for s in args.scenarios if s in SCENARIOS]
    else:
        scenarios = list(SCENARIOS.keys())
    
    # Get technologies
    if args.technologies:
        technologies = args.technologies
    else:
        technologies = EXPECTED_TECHNOLOGIES
    
    print(f"\n🚀 Comprehensive Benchmark Suite")
    print("=" * 80)
    print(f"Technologies: {len(technologies)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Episodes per combination: {episodes}")
    print(f"Total runs: {len(technologies) * len(scenarios) * episodes}")
    print("=" * 80)
    
    # Default environment config
    env_config = {
        "num_lanes": 4,
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40
    }
    
    # Run benchmarks
    all_results = []
    start_time = time.time()
    
    for tech_idx, technology in enumerate(technologies, 1):
        print(f"\n[{tech_idx}/{len(technologies)}] Technology: {technology}")
        print("-" * 80)
        
        for scen_idx, scenario in enumerate(scenarios, 1):
            print(f"  [{scen_idx}/{len(scenarios)}] Scenario: {scenario}")
            
            result = benchmark_technology_scenario(
                technology,
                scenario,
                episodes,
                env_config,
                use_parallel=args.parallel
            )
            
            all_results.append(result)
            
            if result.get("errors", 0) > 0:
                print(f"    ⚠️  Completed with {result['errors']} errors")
            else:
                print(f"    ✅ Avg wait time: {result.get('avg_wait_time', 0):.2f}s")
    
    total_time = time.time() - start_time
    
    # Save results
    output_file = args.output or (PROJECT_ROOT / "results" / "benchmarks" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "episodes": episodes,
            "scenarios": scenarios,
            "technologies": technologies,
            "total_runs": len(technologies) * len(scenarios) * episodes
        },
        "results": all_results,
        "summary": {
            "total_time_seconds": total_time,
            "total_combinations": len(all_results),
            "successful": sum(1 for r in all_results if r.get("errors", 0) == 0),
            "failed": sum(1 for r in all_results if r.get("errors", 0) > 0)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Benchmark complete!")
    print(f"   Results saved to: {output_file}")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Successful: {report['summary']['successful']}/{report['summary']['total_combinations']}")


if __name__ == "__main__":
    import numpy as np
    main()

