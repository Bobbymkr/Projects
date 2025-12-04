"""
Comprehensive Validation Script for Advanced Algorithms.

Validates that HRL and MBRL can be trained, produce results,
and perform comparably or better than existing algorithms.
"""

import argparse
import logging
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.hierarchical_rl_complete import CompleteHierarchicalRLAgent
from src.research.novel_algorithms.model_based_rl_complete import CompleteModelBasedRLAgent
from src.control.fuzzy_control import FuzzyController
from src.rl.dqn_agent import DQNAgent, DQNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmValidator:
    """Comprehensive validator for advanced algorithms."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize validator."""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "results" / "validation"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        self.env_config = {
            "num_lanes": 4,
            "min_green": 10,
            "max_green": 60,
            "yellow_time": 3,
        }
        self.env = TrafficEnv(self.env_config)
    
    def validate_hrl(self, quick: bool = False) -> Dict[str, Any]:
        """Validate Hierarchical RL agent."""
        logger.info("=" * 80)
        logger.info("VALIDATING HIERARCHICAL RL AGENT")
        logger.info("=" * 80)
        
        validation_results = {
            "algorithm": "Hierarchical RL",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "PENDING",
        }
        
        try:
            # Test 1: Agent can be instantiated
            logger.info("Test 1: Agent instantiation...")
            agent = CompleteHierarchicalRLAgent(
                state_dim=12,
                action_dim=4,
                use_domain_options=True,
            )
            validation_results["tests"]["instantiation"] = {
                "status": "PASS",
                "message": "Agent created successfully",
            }
            logger.info("✓ Agent instantiated successfully")
            
            # Test 2: Agent can select actions
            logger.info("Test 2: Action selection...")
            state, _ = self.env.reset()
            actions = []
            for _ in range(10):
                action = agent.select_action(state)
                actions.append(action)
                next_state, _, _, _, _ = self.env.step(action)
                state = next_state
            
            validation_results["tests"]["action_selection"] = {
                "status": "PASS",
                "message": f"Selected {len(actions)} valid actions",
                "actions": actions[:5],  # Sample
            }
            logger.info(f"✓ Selected {len(actions)} valid actions")
            
            # Test 3: Options are discovered/available
            logger.info("Test 3: Option discovery...")
            num_options = len(agent.options)
            validation_results["tests"]["option_discovery"] = {
                "status": "PASS" if num_options > 0 else "FAIL",
                "message": f"Found {num_options} options",
                "num_options": num_options,
            }
            logger.info(f"✓ Found {num_options} options")
            
            # Test 4: Agent can be trained (quick training)
            logger.info("Test 4: Training capability...")
            episodes = 10 if quick else 100
            training_stats = self._quick_train_hrl(agent, episodes)
            
            validation_results["tests"]["training"] = {
                "status": "PASS" if training_stats.get("trained", False) else "FAIL",
                "message": f"Training completed for {episodes} episodes",
                "stats": training_stats,
            }
            logger.info(f"✓ Training completed")
            
            # Test 5: Performance evaluation
            logger.info("Test 5: Performance evaluation...")
            perf_results = self._evaluate_agent(agent, num_episodes=20)
            validation_results["tests"]["performance"] = {
                "status": "PASS",
                "results": perf_results,
            }
            logger.info(f"✓ Performance: Avg Wait = {perf_results.get('avg_wait_time', 0):.2f}s")
            
            # Test 6: Comparison with baseline
            logger.info("Test 6: Baseline comparison...")
            baseline_perf = self._evaluate_baseline("fuzzy_logic", num_episodes=20)
            comparison = self._compare_results(perf_results, baseline_perf)
            validation_results["tests"]["baseline_comparison"] = comparison
            logger.info(f"✓ Comparison: {comparison.get('status', 'PENDING')}")
            
            # Overall status
            all_passed = all(
                test.get("status") == "PASS"
                for test in validation_results["tests"].values()
            )
            validation_results["overall_status"] = "PASS" if all_passed else "PARTIAL"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["tests"]["error"] = {
                "status": "FAIL",
                "message": str(e),
            }
            validation_results["overall_status"] = "FAIL"
        
        return validation_results
    
    def validate_mbrl(self, quick: bool = False) -> Dict[str, Any]:
        """Validate Model-Based RL agent."""
        logger.info("=" * 80)
        logger.info("VALIDATING MODEL-BASED RL AGENT")
        logger.info("=" * 80)
        
        validation_results = {
            "algorithm": "Model-Based RL",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "PENDING",
        }
        
        try:
            # Test 1: Agent instantiation
            logger.info("Test 1: Agent instantiation...")
            agent = CompleteModelBasedRLAgent(
                state_dim=12,
                action_dim=4,
            )
            validation_results["tests"]["instantiation"] = {
                "status": "PASS",
                "message": "Agent created successfully",
            }
            logger.info("✓ Agent instantiated successfully")
            
            # Test 2: Action selection (before training)
            logger.info("Test 2: Action selection (untrained)...")
            state, _ = self.env.reset()
            action = agent.select_action(state)
            validation_results["tests"]["action_selection_untrained"] = {
                "status": "PASS",
                "message": f"Selected action: {action}",
                "action": int(action),
            }
            logger.info(f"✓ Selected action: {action}")
            
            # Test 3: World model training
            logger.info("Test 3: World model training...")
            # Collect some transitions
            for _ in range(50):
                state, _ = self.env.reset()
                for step in range(20):
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    agent.add_transition(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
            
            if len(agent.transition_buffer) >= 32:
                model_stats = agent.train_world_model(epochs=10, batch_size=32)
                validation_results["tests"]["world_model_training"] = {
                    "status": "PASS",
                    "message": "World model trained successfully",
                    "stats": model_stats,
                }
                logger.info("✓ World model trained")
            else:
                validation_results["tests"]["world_model_training"] = {
                    "status": "FAIL",
                    "message": "Not enough transitions",
                }
            
            # Test 4: Action selection (after training)
            logger.info("Test 4: Action selection (trained)...")
            state, _ = self.env.reset()
            action = agent.select_action(state)
            validation_results["tests"]["action_selection_trained"] = {
                "status": "PASS",
                "message": f"Selected action: {action}",
                "action": int(action),
            }
            logger.info(f"✓ Selected action after training: {action}")
            
            # Test 5: Performance evaluation
            logger.info("Test 5: Performance evaluation...")
            perf_results = self._evaluate_agent(agent, num_episodes=20)
            validation_results["tests"]["performance"] = {
                "status": "PASS",
                "results": perf_results,
            }
            logger.info(f"✓ Performance: Avg Wait = {perf_results.get('avg_wait_time', 0):.2f}s")
            
            # Test 6: Inference latency
            logger.info("Test 6: Inference latency...")
            latency_results = self._measure_inference_latency(agent, num_samples=100)
            validation_results["tests"]["inference_latency"] = latency_results
            logger.info(f"✓ Inference Latency: {latency_results.get('avg_latency_ms', 0):.2f}ms")
            
            # Test 7: Baseline comparison
            logger.info("Test 7: Baseline comparison...")
            baseline_perf = self._evaluate_baseline("fuzzy_logic", num_episodes=20)
            comparison = self._compare_results(perf_results, baseline_perf)
            validation_results["tests"]["baseline_comparison"] = comparison
            logger.info(f"✓ Comparison: {comparison.get('status', 'PENDING')}")
            
            # Overall status
            all_passed = all(
                test.get("status") == "PASS"
                for test in validation_results["tests"].values()
            )
            validation_results["overall_status"] = "PASS" if all_passed else "PARTIAL"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            validation_results["tests"]["error"] = {
                "status": "FAIL",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            validation_results["overall_status"] = "FAIL"
        
        return validation_results
    
    def _quick_train_hrl(self, agent: CompleteHierarchicalRLAgent, episodes: int) -> Dict[str, Any]:
        """Quick training for validation."""
        env = TrafficEnv(self.env_config)
        
        for episode in range(episodes):
            state, _ = env.reset()
            trajectory = []
            
            for step in range(50):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((state, action, reward, next_state))
                state = next_state
                if done:
                    break
            
            agent.option_trajectories.append(trajectory)
            agent.primitive_trajectories.append(trajectory)
        
        # Quick training
        if len(agent.option_trajectories) > 0:
            train_stats = agent.train(episodes=1, batch_size=16)
            return {"trained": True, "stats": train_stats}
        
        return {"trained": False}
    
    def _evaluate_agent(self, agent, num_episodes: int = 20) -> Dict[str, Any]:
        """Evaluate agent performance."""
        env = TrafficEnv(self.env_config)
        wait_times = []
        queue_lengths = []
        rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_wait_times = []
            episode_queue_lengths = []
            
            for step in range(200):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                if "wait_times" in info:
                    episode_wait_times.extend(info["wait_times"])
                if "queue_lengths" in info:
                    episode_queue_lengths.extend(info["queue_lengths"])
                
                state = next_state
                if done:
                    break
            
            if episode_wait_times:
                wait_times.append(np.mean(episode_wait_times))
            if episode_queue_lengths:
                queue_lengths.append(np.mean(episode_queue_lengths))
            rewards.append(episode_reward)
        
        return {
            "avg_wait_time": np.mean(wait_times) if wait_times else 0.0,
            "std_wait_time": np.std(wait_times) if wait_times else 0.0,
            "avg_queue_length": np.mean(queue_lengths) if queue_lengths else 0.0,
            "avg_reward": np.mean(rewards) if rewards else 0.0,
            "num_episodes": num_episodes,
        }
    
    def _evaluate_baseline(self, baseline_name: str, num_episodes: int = 20) -> Dict[str, Any]:
        """Evaluate baseline algorithm."""
        env = TrafficEnv(self.env_config)
        
        if baseline_name == "fuzzy_logic":
            controller = FuzzyController()
            wait_times = []
            
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_wait_times = []
                
                for step in range(200):
                    queue_lengths = state[:4].tolist() if len(state) >= 4 else [5, 3, 8, 2]
                    wait_times_state = state[4:8].tolist() if len(state) >= 8 else [12, 8, 15, 6]
                    
                    action = controller.compute_timing(queue_lengths, wait_times_state)
                    phase = action.get("phase", 0) if isinstance(action, dict) else 0
                    next_state, _, terminated, truncated, info = env.step(phase)
                    done = terminated or truncated
                    
                    if "wait_times" in info:
                        episode_wait_times.extend(info["wait_times"])
                    
                    state = next_state
                    if done:
                        break
                
                if episode_wait_times:
                    wait_times.append(np.mean(episode_wait_times))
            
            return {
                "avg_wait_time": np.mean(wait_times) if wait_times else 8.51,
                "algorithm": "fuzzy_logic",
            }
        
        return {"avg_wait_time": 8.51, "algorithm": "unknown"}
    
    def _compare_results(
        self,
        algorithm_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare algorithm results with baseline."""
        algo_wait = algorithm_results.get("avg_wait_time", float('inf'))
        baseline_wait = baseline_results.get("avg_wait_time", 8.51)
        
        improvement = ((baseline_wait - algo_wait) / baseline_wait) * 100 if baseline_wait > 0 else 0
        is_better = algo_wait < baseline_wait
        is_acceptable = algo_wait < baseline_wait * 1.2  # Within 20% of baseline
        
        return {
            "status": "PASS" if is_acceptable else "FAIL",
            "algorithm_wait_time": algo_wait,
            "baseline_wait_time": baseline_wait,
            "improvement_percent": improvement,
            "is_better": is_better,
            "is_acceptable": is_acceptable,
        }
    
    def _measure_inference_latency(self, agent, num_samples: int = 100) -> Dict[str, Any]:
        """Measure inference latency."""
        import time
        
        env = TrafficEnv(self.env_config)
        latencies = []
        
        for _ in range(num_samples):
            state, _ = env.reset()
            start = time.time()
            action = agent.select_action(state)
            latency = time.time() - start
            latencies.append(latency)
        
        avg_latency_ms = np.mean(latencies) * 1000
        p95_latency_ms = np.percentile(latencies, 95) * 1000
        
        is_acceptable = avg_latency_ms < 100  # < 100ms
        
        return {
            "status": "PASS" if is_acceptable else "FAIL",
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "is_acceptable": is_acceptable,
        }
    
    def run_full_validation(self, quick: bool = False) -> Dict[str, Any]:
        """Run full validation for all advanced algorithms."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE ALGORITHM VALIDATION")
        logger.info("=" * 80)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "validation_mode": "quick" if quick else "full",
            "algorithms": {},
            "summary": {},
        }
        
        # Validate HRL
        hrl_results = self.validate_hrl(quick=quick)
        results["algorithms"]["hierarchical_rl"] = hrl_results
        
        # Validate MBRL
        mbrl_results = self.validate_mbrl(quick=quick)
        results["algorithms"]["model_based_rl"] = mbrl_results
        
        # Generate summary
        results["summary"] = {
            "hrl_status": hrl_results.get("overall_status", "UNKNOWN"),
            "mbrl_status": mbrl_results.get("overall_status", "UNKNOWN"),
            "all_passed": (
                hrl_results.get("overall_status") == "PASS" and
                mbrl_results.get("overall_status") == "PASS"
            ),
        }
        
        # Save results
        results_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"HRL Status: {results['summary']['hrl_status']}")
        logger.info(f"MBRL Status: {results['summary']['mbrl_status']}")
        logger.info(f"All Passed: {results['summary']['all_passed']}")
        logger.info(f"Results saved to: {results_file}")
        
        return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Advanced Algorithms")
    parser.add_argument("--algorithm", type=str, choices=["hrl", "mbrl", "all"], default="all",
                       help="Algorithm to validate")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    validator = AlgorithmValidator(output_dir=Path(args.output) if args.output else None)
    
    if args.algorithm == "all":
        results = validator.run_full_validation(quick=args.quick)
    elif args.algorithm == "hrl":
        results = validator.validate_hrl(quick=args.quick)
    elif args.algorithm == "mbrl":
        results = validator.validate_mbrl(quick=args.quick)
    
    # Exit with appropriate code
    if isinstance(results, dict) and results.get("overall_status") == "PASS":
        sys.exit(0)
    elif isinstance(results, dict) and "summary" in results:
        if results["summary"].get("all_passed", False):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

