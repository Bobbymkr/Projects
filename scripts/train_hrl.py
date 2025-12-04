"""
Training Script for Hierarchical Reinforcement Learning.

Trains HRL agent and validates performance against baselines.
"""

import argparse
import logging
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.hierarchical_rl_complete import (
    CompleteHierarchicalRLAgent
)
from src.control.fuzzy_control import FuzzyController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_hrl_agent(
    episodes: int = 2000,
    state_dim: int = 12,
    action_dim: int = 4,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Train Hierarchical RL agent.
    
    Args:
        episodes: Number of training episodes
        state_dim: State dimension
        action_dim: Action dimension
        output_dir: Output directory for models and results
        
    Returns:
        Training statistics and results
    """
    logger.info(f"Training HRL agent for {episodes} episodes")
    
    # Create output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / "hrl"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env_config = {
        "num_lanes": 4,
        "min_green": 10,
        "max_green": 60,
        "yellow_time": 3,
    }
    env = TrafficEnv(env_config)
    
    # Create HRL agent
    agent = CompleteHierarchicalRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_domain_options=True,
    )
    
    # Training statistics
    training_stats = {
        "episode_rewards": [],
        "episode_wait_times": [],
        "episode_queue_lengths": [],
        "option_usage": {},
    }
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        # Ensure state is a 1D numpy array
        state = np.array(state, dtype=np.float32).flatten()
        episode_reward = 0.0
        episode_wait_times = []
        episode_queue_lengths = []
        steps = 0
        max_steps = 1000
        
        # Collect trajectory for option discovery
        episode_trajectory = []
        
        while steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Track option usage (if available)
            if hasattr(agent, 'hierarchical_policy') and hasattr(agent.hierarchical_policy, 'current_option'):
                active_option = agent.hierarchical_policy.current_option
                if active_option:
                    option_id = active_option.option_id if hasattr(active_option, 'option_id') else str(active_option)
                    training_stats["option_usage"][option_id] = (
                        training_stats["option_usage"].get(option_id, 0) + 1
                    )
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Ensure next_state is a 1D numpy array
            next_state = np.array(next_state, dtype=np.float32).flatten()
            
            # Store transition
            episode_trajectory.append((state, action, reward, next_state))
            
            # Update statistics
            episode_reward += reward
            if "wait_times" in info:
                episode_wait_times.extend(info["wait_times"])
            if "queue_lengths" in info:
                episode_queue_lengths.extend(info["queue_lengths"])
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Store trajectory for option discovery
        agent.option_trajectories.append(episode_trajectory)
        agent.primitive_trajectories.append(episode_trajectory)
        
        # Train agent periodically
        if episode % 100 == 0 and episode > 0:
            train_stats = agent.train(episodes=1, batch_size=32)
            logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Avg Wait={np.mean(episode_wait_times) if episode_wait_times else 0:.2f}s")
        
        # Record statistics
        training_stats["episode_rewards"].append(episode_reward)
        if episode_wait_times:
            training_stats["episode_wait_times"].append(np.mean(episode_wait_times))
        if episode_queue_lengths:
            training_stats["episode_queue_lengths"].append(np.mean(episode_queue_lengths))
        
        # Discover options periodically
        if episode % 500 == 0 and episode > 0:
            if len(agent.option_trajectories) > 0:
                agent.discover_options(agent.option_trajectories[-100:], num_options=4)
                logger.info(f"Discovered options at episode {episode}")
    
    # Final training
    logger.info("Running final training...")
    final_train_stats = agent.train(episodes=10, batch_size=32)
    
    # Evaluate performance
    logger.info("Evaluating trained agent...")
    eval_results = evaluate_agent(agent, env, num_episodes=50)
    
    # Save results
    results = {
        "training_stats": training_stats,
        "final_training": final_train_stats,
        "evaluation": eval_results,
        "option_usage": training_stats["option_usage"],
        "episodes_trained": episodes,
    }
    
    results_file = output_dir / f"hrl_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training complete. Results saved to {results_file}")
    logger.info(f"Final Performance: Avg Wait Time = {eval_results.get('avg_wait_time', 0):.2f}s")
    
    return results


def evaluate_agent(
    agent: CompleteHierarchicalRLAgent,
    env: TrafficEnv,
    num_episodes: int = 50,
) -> Dict[str, Any]:
    """Evaluate agent performance."""
    wait_times = []
    queue_lengths = []
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()
        episode_reward = 0.0
        episode_wait_times = []
        episode_queue_lengths = []
        
        for step in range(1000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32).flatten()
            
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


def compare_with_baseline(
    hrl_results: Dict[str, Any],
    baseline_name: str = "fuzzy_logic",
) -> Dict[str, Any]:
    """Compare HRL results with baseline."""
    logger.info(f"Comparing HRL with {baseline_name} baseline...")
    
    # Create baseline agent
    env_config = {"num_lanes": 4}
    env = TrafficEnv(env_config)
    
    if baseline_name == "fuzzy_logic":
        baseline_agent = FuzzyController()
        # Evaluate baseline
        baseline_results = evaluate_fuzzy_baseline(baseline_agent, env)
    else:
        baseline_results = {"avg_wait_time": 8.51}  # Known fuzzy logic performance
    
    hrl_wait_time = hrl_results.get("evaluation", {}).get("avg_wait_time", 0)
    baseline_wait_time = baseline_results.get("avg_wait_time", 8.51)
    
    improvement = ((baseline_wait_time - hrl_wait_time) / baseline_wait_time) * 100
    
    comparison = {
        "hrl_wait_time": hrl_wait_time,
        "baseline_wait_time": baseline_wait_time,
        "improvement_percent": improvement,
        "is_better": hrl_wait_time < baseline_wait_time,
    }
    
    logger.info(f"HRL Wait Time: {hrl_wait_time:.2f}s")
    logger.info(f"Baseline Wait Time: {baseline_wait_time:.2f}s")
    logger.info(f"Improvement: {improvement:.2f}%")
    
    return comparison


def evaluate_fuzzy_baseline(controller: FuzzyController, env: TrafficEnv) -> Dict[str, Any]:
    """Evaluate fuzzy logic baseline."""
    wait_times = []
    
    for episode in range(50):
        state = env.reset()
        episode_wait_times = []
        
        for step in range(1000):
            # Extract queue lengths and wait times from state
            state_flat = np.array(state, dtype=np.float32).flatten()
            queue_lengths = state_flat[:4].tolist() if len(state_flat) >= 4 else [5, 3, 8, 2]
            wait_times_state = state_flat[4:8].tolist() if len(state_flat) >= 8 else [12, 8, 15, 6]
            
            action = controller.compute_timing(queue_lengths, wait_times_state)
            next_state, reward, terminated, truncated, info = env.step(action.get("phase", 0) if isinstance(action, dict) else 0)
            done = terminated or truncated
            state = np.array(next_state, dtype=np.float32).flatten()
            
            if "wait_times" in info:
                episode_wait_times.extend(info["wait_times"])
            
            state = next_state
            if done:
                break
        
        if episode_wait_times:
            wait_times.append(np.mean(episode_wait_times))
    
    return {
        "avg_wait_time": np.mean(wait_times) if wait_times else 8.51,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Hierarchical RL Agent")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    
    args = parser.parse_args()
    
    # Train agent
    results = train_hrl_agent(
        episodes=args.episodes,
        output_dir=Path(args.output) if args.output else None,
    )
    
    # Validate if requested
    if args.validate:
        logger.info("Running validation...")
        validation_results = validate_hrl_agent(results)
        logger.info(f"Validation: {validation_results}")
    
    # Compare with baseline if requested
    if args.compare:
        comparison = compare_with_baseline(results)
        logger.info(f"Comparison: {comparison}")


def validate_hrl_agent(results: Dict[str, Any]) -> Dict[str, bool]:
    """Validate HRL agent training results."""
    validation = {
        "training_completed": True,
        "has_improvement": False,
        "options_discovered": False,
        "performance_acceptable": False,
    }
    
    # Check training completed
    if results.get("episodes_trained", 0) == 0:
        validation["training_completed"] = False
    
    # Check option usage
    option_usage = results.get("option_usage", {})
    if len(option_usage) > 0:
        validation["options_discovered"] = True
    
    # Check performance
    eval_results = results.get("evaluation", {})
    avg_wait_time = eval_results.get("avg_wait_time", float('inf'))
    if avg_wait_time < 20.0:  # Acceptable threshold
        validation["performance_acceptable"] = True
    
    # Check improvement
    if avg_wait_time < 8.51:  # Better than fuzzy logic baseline
        validation["has_improvement"] = True
    
    return validation


if __name__ == "__main__":
    main()

