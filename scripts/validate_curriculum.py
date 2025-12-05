"""
Curriculum Learning Validation Script.

Compares training with and without curriculum learning to validate improvements.
Implements Phase 2.1 validation from SCORE_IMPROVEMENT_ROADMAP.md.
"""

import json
import os
import argparse
import numpy as np
from tqdm import trange
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.curriculum_learning import TrafficCurriculum
from src.rl.convergence_monitor import ConvergenceMonitor


def load_config(path: str):
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def train_with_curriculum(cfg_path: str, episodes: int, use_curriculum: bool = True):
    """
    Train agent with or without curriculum learning.
    
    Returns:
        Dictionary with training metrics
    """
    env_cfg = load_config(cfg_path)
    env = TrafficEnv(env_cfg)
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=DQNConfig()
    )
    
    # Initialize curriculum if enabled
    curriculum = None
    if use_curriculum:
        base_arrival_rates = env_cfg.get("arrival_rates", [0.3] * env.num_lanes)
        curriculum = TrafficCurriculum(
            base_arrival_rates=base_arrival_rates,
            performance_threshold=0.7,
            min_episodes_per_level=50,
            performance_window=100
        )
    
    convergence_monitor = ConvergenceMonitor(
        window=100,
        threshold=0.01,
        patience=500,
        min_episodes=200,
        mode="maximize"
    )
    
    rewards = []
    episode_lengths = []
    curriculum_levels = []
    
    for ep in trange(episodes, desc=f"Training {'with' if use_curriculum else 'without'} curriculum"):
        # Update environment with curriculum level
        if curriculum is not None:
            current_level = curriculum.get_current_level()
            env.arrival_rates = curriculum.get_arrival_rates()
            curriculum_levels.append(current_level.level_id)
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.select_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(
                obs.astype(np.float32),
                action,
                reward,
                next_obs.astype(np.float32),
                terminated or truncated
            )
            loss = agent.train_step()
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Update curriculum
        if curriculum is not None:
            curriculum.update_performance(episode_reward, ep)
        
        # Update convergence monitor
        convergence_monitor.update(episode_reward, ep)
    
    return {
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "curriculum_levels": curriculum_levels if curriculum else None,
        "final_curriculum_level": curriculum.current_level if curriculum else None,
        "convergence_episode": convergence_monitor.best_episode,
    }


def validate_curriculum(cfg_path: str, episodes: int, num_runs: int = 3):
    """
    Validate curriculum learning by comparing with and without curriculum.
    
    Args:
        cfg_path: Path to environment config
        episodes: Number of episodes per run
        num_runs: Number of independent runs for statistical significance
    
    Returns:
        Comparison results dictionary
    """
    print("=" * 60)
    print("Curriculum Learning Validation")
    print("=" * 60)
    print(f"Episodes per run: {episodes}")
    print(f"Number of runs: {num_runs}")
    print()
    
    results_with_curriculum = []
    results_without_curriculum = []
    
    # Run with curriculum
    print("Running with curriculum learning...")
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs} (with curriculum)")
        result = train_with_curriculum(cfg_path, episodes, use_curriculum=True)
        results_with_curriculum.append(result)
    
    # Run without curriculum
    print("\n" + "=" * 60)
    print("Running without curriculum learning...")
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs} (without curriculum)")
        result = train_with_curriculum(cfg_path, episodes, use_curriculum=False)
        results_without_curriculum.append(result)
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Results Analysis")
    print("=" * 60)
    
    # Calculate metrics
    def calculate_metrics(results_list):
        all_final_rewards = [np.mean(r["rewards"][-100:]) for r in results_list]
        all_convergence_episodes = [r["convergence_episode"] for r in results_list if r["convergence_episode"]]
        all_episode_lengths = [np.mean(r["episode_lengths"]) for r in results_list]
        
        return {
            "mean_final_reward": np.mean(all_final_rewards),
            "std_final_reward": np.std(all_final_rewards),
            "mean_convergence_episode": np.mean(all_convergence_episodes) if all_convergence_episodes else None,
            "mean_episode_length": np.mean(all_episode_lengths),
        }
    
    metrics_with = calculate_metrics(results_with_curriculum)
    metrics_without = calculate_metrics(results_without_curriculum)
    
    # Print comparison
    print("\nFinal Performance (last 100 episodes):")
    print(f"  With Curriculum:    {metrics_with['mean_final_reward']:.2f} ± {metrics_with['std_final_reward']:.2f}")
    print(f"  Without Curriculum: {metrics_without['mean_final_reward']:.2f} ± {metrics_without['std_final_reward']:.2f}")
    improvement = ((metrics_with['mean_final_reward'] - metrics_without['mean_final_reward']) / 
                   abs(metrics_without['mean_final_reward']) * 100)
    print(f"  Improvement:        {improvement:+.1f}%")
    
    print("\nConvergence Speed:")
    if metrics_with['mean_convergence_episode'] and metrics_without['mean_convergence_episode']:
        print(f"  With Curriculum:    Episode {metrics_with['mean_convergence_episode']:.0f}")
        print(f"  Without Curriculum: Episode {metrics_without['mean_convergence_episode']:.0f}")
        speedup = ((metrics_without['mean_convergence_episode'] - metrics_with['mean_convergence_episode']) / 
                   metrics_without['mean_convergence_episode'] * 100)
        print(f"  Speedup:            {speedup:+.1f}%")
    
    print("\nAverage Episode Length:")
    print(f"  With Curriculum:    {metrics_with['mean_episode_length']:.1f} steps")
    print(f"  Without Curriculum: {metrics_without['mean_episode_length']:.1f} steps")
    
    # Curriculum progression
    if results_with_curriculum[0]["curriculum_levels"]:
        final_levels = [r["final_curriculum_level"] for r in results_with_curriculum]
        print(f"\nCurriculum Progression:")
        print(f"  Final level reached: {np.mean(final_levels):.1f} ± {np.std(final_levels):.1f}")
        print(f"  Max level: {max(final_levels)}")
    
    return {
        "with_curriculum": metrics_with,
        "without_curriculum": metrics_without,
        "improvement_percent": improvement,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate curriculum learning")
    parser.add_argument("--config", default="configs/intersection.json", help="Environment config")
    parser.add_argument("--episodes", type=int, default=300, help="Episodes per run")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--out", default="runs/curriculum_validation", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    results = validate_curriculum(args.config, args.episodes, args.runs)
    
    # Save results
    import json
    results_path = os.path.join(args.out, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

