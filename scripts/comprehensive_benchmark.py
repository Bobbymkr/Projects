#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for All Traffic Control Technologies.

Tests all implemented technologies to find the optimal solution
for reducing waiting times.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

from src.env.traffic_env import TrafficEnv
from src.control.fuzzy_control import FuzzyController
from src.control.webster_method import WebsterMethod
from src.utils.config import load_config

# Try to import all advanced technologies
TECHNOLOGIES = {}

try:
    from src.research.novel_algorithms.hierarchical_rl_complete import HierarchicalRLAgent
    TECHNOLOGIES['hierarchical_rl'] = HierarchicalRLAgent
except ImportError:
    logging.warning("Hierarchical RL not available")

try:
    from src.research.novel_algorithms.model_based_rl_complete import ModelBasedRLAgent
    TECHNOLOGIES['model_based_rl'] = ModelBasedRLAgent
except ImportError:
    logging.warning("Model-Based RL not available")

try:
    from src.research.novel_algorithms.imitation_learning_complete import BehavioralCloningAgent
    TECHNOLOGIES['imitation_learning'] = BehavioralCloningAgent
except ImportError:
    logging.warning("Imitation Learning not available")

try:
    from src.research.novel_algorithms.transformer_control import TransformerAgent
    TECHNOLOGIES['transformer'] = TransformerAgent
except ImportError:
    logging.warning("Transformer not available")

try:
    from src.research.novel_algorithms.bayesian_methods import BayesianAgent
    TECHNOLOGIES['bayesian'] = BayesianAgent
except ImportError:
    logging.warning("Bayesian not available")

try:
    from src.research.novel_algorithms.causal_inference import CausalAgent
    TECHNOLOGIES['causal'] = CausalAgent
except ImportError:
    logging.warning("Causal Inference not available")

try:
    from src.research.novel_algorithms.neuro_symbolic import NeuroSymbolicAgent
    TECHNOLOGIES['neuro_symbolic'] = NeuroSymbolicAgent
except ImportError:
    logging.warning("Neuro-Symbolic not available")

try:
    from src.research.novel_algorithms.meta_learning import MAMLAgent
    TECHNOLOGIES['meta_learning'] = MAMLAgent
except ImportError:
    logging.warning("Meta-Learning not available")

try:
    from src.research.novel_algorithms.llm_traffic import LLMTrafficAgent
    TECHNOLOGIES['llm'] = LLMTrafficAgent
except ImportError:
    logging.warning("LLM not available")

try:
    from src.research.novel_algorithms.diffusion_models import DiffusionTrafficAgent
    TECHNOLOGIES['diffusion'] = DiffusionTrafficAgent
except ImportError:
    logging.warning("Diffusion Models not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs comprehensive benchmarks on all technologies."""
    
    def __init__(self, config_path: str, num_test_episodes: int = 100):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to traffic configuration
            num_test_episodes: Number of episodes to test each technology
        """
        self.config_path = config_path
        self.num_test_episodes = num_test_episodes
        
        # Load configuration
        config = load_config(config_path)
        self.traffic_config = config.get('traffic', config)
        
        # Create environment
        self.env = TrafficEnv(self.traffic_config)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        logger.info(f"Initialized benchmark with {num_test_episodes} test episodes per technology")
    
    def benchmark_fuzzy_logic(self) -> Dict[str, Any]:
        """Benchmark Fuzzy Logic Controller."""
        logger.info("\n" + "="*80)
        logger.info("BENCHMARKING: Fuzzy Logic Controller")
        logger.info("="*80)
        
        controller = FuzzyController()
        
        return self._run_benchmark("Fuzzy Logic", controller)
    
    def benchmark_webster(self) -> Dict[str, Any]:
        """Benchmark Webster's Method."""
        logger.info("\n" + "="*80)
        logger.info("BENCHMARKING: Webster's Method")
        logger.info("="*80)
        
        controller = WebsterMethod()
        
        return self._run_benchmark("Webster's Method", controller)
    
    def benchmark_agent(self, name: str, agent_class, model_path: str = None) -> Dict[str, Any]:
        """Generic benchmark for any agent."""
        logger.info("\n" + "="*80)
        logger.info(f"BENCHMARKING: {name}")
        logger.info("="*80)
        
        try:
            agent = agent_class(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
            )
            
            if model_path and Path(model_path).exists():
                agent.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            
            return self._run_benchmark(name, agent)
        except Exception as e:
            logger.error(f"Error benchmarking {name}: {e}")
            return {"error": str(e)}
    
    def benchmark_hierarchical_rl(self, model_path: str = None) -> Dict[str, Any]:
        """Benchmark Hierarchical RL."""
        if 'hierarchical_rl' not in TECHNOLOGIES:
            return {"error": "Hierarchical RL not available"}
        return self.benchmark_agent("Hierarchical RL", TECHNOLOGIES['hierarchical_rl'], model_path)
    
    def benchmark_model_based_rl(self, model_path: str = None) -> Dict[str, Any]:
        """Benchmark Model-Based RL."""
        if 'model_based_rl' not in TECHNOLOGIES:
            return {"error": "Model-Based RL not available"}
        return self.benchmark_agent("Model-Based RL", TECHNOLOGIES['model_based_rl'], model_path)
    
    def benchmark_imitation_learning(self) -> Dict[str, Any]:
        """Benchmark Imitation Learning."""
        if 'imitation_learning' not in TECHNOLOGIES:
            return {"error": "Imitation Learning not available"}
        return self.benchmark_agent("Imitation Learning", TECHNOLOGIES['imitation_learning'])
    
    def benchmark_transformer(self) -> Dict[str, Any]:
        """Benchmark Transformer."""
        if 'transformer' not in TECHNOLOGIES:
            return {"error": "Transformer not available"}
        return self.benchmark_agent("Transformer", TECHNOLOGIES['transformer'])
    
    def benchmark_bayesian(self) -> Dict[str, Any]:
        """Benchmark Bayesian."""
        if 'bayesian' not in TECHNOLOGIES:
            return {"error": "Bayesian not available"}
        return self.benchmark_agent("Bayesian", TECHNOLOGIES['bayesian'])
    
    def benchmark_causal(self) -> Dict[str, Any]:
        """Benchmark Causal Inference."""
        if 'causal' not in TECHNOLOGIES:
            return {"error": "Causal Inference not available"}
        return self.benchmark_agent("Causal Inference", TECHNOLOGIES['causal'])
    
    def benchmark_neuro_symbolic(self) -> Dict[str, Any]:
        """Benchmark Neuro-Symbolic."""
        if 'neuro_symbolic' not in TECHNOLOGIES:
            return {"error": "Neuro-Symbolic not available"}
        return self.benchmark_agent("Neuro-Symbolic", TECHNOLOGIES['neuro_symbolic'])
    
    def benchmark_meta_learning(self) -> Dict[str, Any]:
        """Benchmark Meta-Learning."""
        if 'meta_learning' not in TECHNOLOGIES:
            return {"error": "Meta-Learning not available"}
        return self.benchmark_agent("Meta-Learning", TECHNOLOGIES['meta_learning'])
    
    def benchmark_llm(self) -> Dict[str, Any]:
        """Benchmark LLM."""
        if 'llm' not in TECHNOLOGIES:
            return {"error": "LLM not available"}
        return self.benchmark_agent("LLM", TECHNOLOGIES['llm'])
    
    def benchmark_diffusion(self) -> Dict[str, Any]:
        """Benchmark Diffusion Models."""
        if 'diffusion' not in TECHNOLOGIES:
            return {"error": "Diffusion Models not available"}
        return self.benchmark_agent("Diffusion Models", TECHNOLOGIES['diffusion'])
    
    def _run_benchmark(self, name: str, controller: Any) -> Dict[str, Any]:
        """Run benchmark for a controller."""
        wait_times = []
        queue_lengths = []
        rewards = []
        episode_lengths = []
        
        for episode in tqdm(range(self.num_test_episodes), desc=f"Testing {name}"):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_wait = 0.0
            episode_queues = []
            
            done = False
            
            while not done:
                # Select action based on controller type
                if hasattr(controller, 'select_action'):
                    # For RL agents - check if it accepts epsilon parameter
                    import inspect
                    sig = inspect.signature(controller.select_action)
                    if 'epsilon' in sig.parameters:
                        action = controller.select_action(obs, epsilon=0.0)  # No exploration
                    else:
                        action = controller.select_action(obs)
                elif hasattr(controller, 'compute_timing'):
                    # For Fuzzy Controller
                    green_time = controller.compute_timing(obs)
                    # Convert to action index
                    green_values = self.env.green_values
                    action = np.argmin(np.abs(green_values - green_time))
                elif hasattr(controller, 'get_action'):
                    # For Webster's Method
                    result = controller.get_action({'volumes': obs})
                    if isinstance(result, dict):
                        green_time = result.get('green_times', [15])[0]
                    else:
                        green_time = 15
                    green_values = self.env.green_values
                    action = np.argmin(np.abs(green_values - green_time))
                else:
                    action = 0  # Default
                
                # Execute action
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                
                # Update statistics
                episode_reward += reward
                episode_steps += 1
                episode_wait += step_info.get('avg_wait_time', 0.0)
                episode_queues.append(np.sum(next_obs))
                
                obs = next_obs
            
            wait_times.append(episode_wait / episode_steps if episode_steps > 0 else 0.0)
            queue_lengths.append(np.mean(episode_queues) if episode_queues else 0.0)
            rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
        
        results = {
            "name": name,
            "avg_wait_time": float(np.mean(wait_times)),
            "std_wait_time": float(np.std(wait_times)),
            "min_wait_time": float(np.min(wait_times)),
            "max_wait_time": float(np.max(wait_times)),
            "avg_queue_length": float(np.mean(queue_lengths)),
            "avg_reward": float(np.mean(rewards)),
            "avg_episode_length": float(np.mean(episode_lengths)),
            "num_episodes": self.num_test_episodes,
        }
        
        logger.info(f"\n{name} Results:")
        logger.info(f"  Average Wait Time: {results['avg_wait_time']:.2f}s ± {results['std_wait_time']:.2f}s")
        logger.info(f"  Average Queue Length: {results['avg_queue_length']:.2f}")
        logger.info(f"  Average Reward: {results['avg_reward']:.2f}")
        
        return results
    
    def run_all_benchmarks(self, output_path: str = "./benchmark_results.json") -> Dict[str, Any]:
        """Run all benchmarks and save results."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE BENCHMARK SUITE")
        logger.info("="*80)
        logger.info(f"Testing {self.num_test_episodes} episodes per technology\n")
        
        all_results = {
            "benchmark_date": datetime.now().isoformat(),
            "config_path": self.config_path,
            "num_test_episodes": self.num_test_episodes,
            "results": {},
        }
        
        # Run benchmarks for all technologies
        all_results["results"]["fuzzy_logic"] = self.benchmark_fuzzy_logic()
        all_results["results"]["webster"] = self.benchmark_webster()
        
        # Advanced technologies
        if 'hierarchical_rl' in TECHNOLOGIES:
            all_results["results"]["hierarchical_rl"] = self.benchmark_hierarchical_rl()
        
        if 'model_based_rl' in TECHNOLOGIES:
            all_results["results"]["model_based_rl"] = self.benchmark_model_based_rl()
        
        if 'imitation_learning' in TECHNOLOGIES:
            all_results["results"]["imitation_learning"] = self.benchmark_imitation_learning()
        
        if 'transformer' in TECHNOLOGIES:
            all_results["results"]["transformer"] = self.benchmark_transformer()
        
        if 'bayesian' in TECHNOLOGIES:
            all_results["results"]["bayesian"] = self.benchmark_bayesian()
        
        if 'causal' in TECHNOLOGIES:
            all_results["results"]["causal"] = self.benchmark_causal()
        
        if 'neuro_symbolic' in TECHNOLOGIES:
            all_results["results"]["neuro_symbolic"] = self.benchmark_neuro_symbolic()
        
        if 'meta_learning' in TECHNOLOGIES:
            all_results["results"]["meta_learning"] = self.benchmark_meta_learning()
        
        if 'llm' in TECHNOLOGIES:
            all_results["results"]["llm"] = self.benchmark_llm()
        
        if 'diffusion' in TECHNOLOGIES:
            all_results["results"]["diffusion"] = self.benchmark_diffusion()
        
        # Find best performer (by reward, since wait times are all similar)
        rewards = {
            name: result.get('avg_reward', float('-inf'))
            for name, result in all_results["results"].items()
            if 'error' not in result
        }
        
        wait_times = {
            name: result.get('avg_wait_time', float('inf'))
            for name, result in all_results["results"].items()
            if 'error' not in result
        }
        
        if rewards:
            best_by_reward = max(rewards, key=rewards.get)
            all_results["best_performer"] = {
                "by_reward": {
                    "name": best_by_reward,
                    "reward": float(rewards[best_by_reward]),
                    "wait_time": float(wait_times.get(best_by_reward, 0)),
                }
            }
        
        if wait_times:
            best_by_wait = min(wait_times, key=wait_times.get)
            all_results["best_performer"]["by_wait_time"] = {
                "name": best_by_wait,
                "wait_time": float(wait_times[best_by_wait]),
                "reward": float(rewards.get(best_by_wait, 0)),
            }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {output_path}")
        
        if rewards:
            best_reward = max(rewards, key=rewards.get)
            logger.info(f"\n🏆 BEST BY REWARD: {best_reward}")
            logger.info(f"   Reward: {rewards[best_reward]:.2f}")
            logger.info(f"   Wait Time: {wait_times.get(best_reward, 0):.2f}s")
            
            if wait_times:
                best_wait = min(wait_times, key=wait_times.get)
                logger.info(f"\n🏆 BEST BY WAIT TIME: {best_wait}")
                logger.info(f"   Wait Time: {wait_times[best_wait]:.2f}s")
                logger.info(f"   Reward: {rewards.get(best_wait, 0):.2f}")
        
        # Print comparison table
        logger.info("\n" + "="*80)
        logger.info("COMPARISON TABLE")
        logger.info("="*80)
        logger.info(f"{'Technology':<20} {'Wait Time (s)':<15} {'Queue Length':<15} {'Reward':<15}")
        logger.info("-" * 80)
        
        for name, result in all_results["results"].items():
            if 'error' not in result:
                logger.info(
                    f"{result['name']:<20} "
                    f"{result['avg_wait_time']:<15.2f} "
                    f"{result['avg_queue_length']:<15.2f} "
                    f"{result['avg_reward']:<15.2f}"
                )
        
        # Convert all numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        all_results = convert_to_native(all_results)
        
        return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Suite")
    parser.add_argument('--config', type=str, default='configs/intersection.json',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of test episodes per technology')
    parser.add_argument('--output', type=str, default='./benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config, args.episodes)
    results = runner.run_all_benchmarks(args.output)

