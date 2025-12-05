"""
Performance Benchmarking Script.

Benchmarks PER vs uniform replay, distributional RL, and curriculum learning.
Implements Phase 3.3 from Expert Review Remediation Plan.
"""

import json
import os
import argparse
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.rl.curriculum_learning import TrafficCurriculum
from src.rl.convergence_monitor import ConvergenceMonitor


def load_config(path: str):
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def benchmark_per_vs_uniform(cfg_path: str, episodes: int = 200):
    """
    Benchmark PER vs uniform replay.
    
    Returns:
        Dictionary with performance metrics
    """
    env_cfg = load_config(cfg_path)
    env = TrafficEnv(env_cfg)
    
    results = {}
    
    # Test uniform replay
    print("Benchmarking uniform replay...")
    cfg_uniform = DQNConfig()
    cfg_uniform.use_per = False
    agent_uniform = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=cfg_uniform
    )
    
    start_time = time.time()
    rewards_uniform = []
    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent_uniform.select_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent_uniform.push(
                obs.astype(np.float32),
                action,
                reward,
                next_obs.astype(np.float32),
                terminated or truncated
            )
            agent_uniform.train_step()
            episode_reward += reward
            obs = next_obs
        rewards_uniform.append(episode_reward)
    uniform_time = time.time() - start_time
    
    results["uniform"] = {
        "final_reward": np.mean(rewards_uniform[-50:]),
        "std_reward": np.std(rewards_uniform[-50:]),
        "training_time": uniform_time,
        "convergence_episode": None,  # Would need convergence monitor
    }
    
    # Test PER
    print("Benchmarking PER...")
    env = TrafficEnv(env_cfg)  # Reset environment
    cfg_per = DQNConfig()
    cfg_per.use_per = True
    cfg_per.per_alpha = 0.6
    cfg_per.per_beta = 0.4
    agent_per = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=cfg_per
    )
    
    start_time = time.time()
    rewards_per = []
    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent_per.select_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent_per.push(
                obs.astype(np.float32),
                action,
                reward,
                next_obs.astype(np.float32),
                terminated or truncated
            )
            agent_per.train_step()
            episode_reward += reward
            obs = next_obs
        rewards_per.append(episode_reward)
    per_time = time.time() - start_time
    
    results["per"] = {
        "final_reward": np.mean(rewards_per[-50:]),
        "std_reward": np.std(rewards_per[-50:]),
        "training_time": per_time,
        "convergence_episode": None,
    }
    
    # Calculate improvement
    reward_improvement = ((results["per"]["final_reward"] - results["uniform"]["final_reward"]) /
                         abs(results["uniform"]["final_reward"]) * 100)
    time_overhead = ((per_time - uniform_time) / uniform_time * 100)
    
    results["comparison"] = {
        "reward_improvement_percent": reward_improvement,
        "time_overhead_percent": time_overhead,
    }
    
    return results


def benchmark_curriculum_learning(cfg_path: str, episodes: int = 200):
    """
    Benchmark curriculum learning vs fixed difficulty.
    
    Returns:
        Dictionary with performance metrics
    """
    env_cfg = load_config(cfg_path)
    
    results = {}
    
    # Test without curriculum
    print("Benchmarking without curriculum...")
    env = TrafficEnv(env_cfg)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=DQNConfig()
    )
    
    start_time = time.time()
    rewards_no_curriculum = []
    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
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
            agent.train_step()
            episode_reward += reward
            obs = next_obs
        rewards_no_curriculum.append(episode_reward)
    no_curriculum_time = time.time() - start_time
    
    results["no_curriculum"] = {
        "final_reward": np.mean(rewards_no_curriculum[-50:]),
        "std_reward": np.std(rewards_no_curriculum[-50:]),
        "training_time": no_curriculum_time,
    }
    
    # Test with curriculum
    print("Benchmarking with curriculum...")
    env = TrafficEnv(env_cfg)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=DQNConfig()
    )
    
    base_arrival_rates = env_cfg.get("arrival_rates", [0.3] * env.num_lanes)
    curriculum = TrafficCurriculum(
        base_arrival_rates=base_arrival_rates,
        performance_threshold=0.7,
        min_episodes_per_level=20,
        performance_window=50,
    )
    
    start_time = time.time()
    rewards_curriculum = []
    for ep in range(episodes):
        env.arrival_rates = curriculum.get_arrival_rates()
        obs, info = env.reset()
        episode_reward = 0.0
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
            agent.train_step()
            episode_reward += reward
            obs = next_obs
        rewards_curriculum.append(episode_reward)
        curriculum.update_performance(episode_reward, ep)
    curriculum_time = time.time() - start_time
    
    results["curriculum"] = {
        "final_reward": np.mean(rewards_curriculum[-50:]),
        "std_reward": np.std(rewards_curriculum[-50:]),
        "training_time": curriculum_time,
        "final_level": curriculum.current_level,
    }
    
    # Calculate improvement
    reward_improvement = ((results["curriculum"]["final_reward"] - results["no_curriculum"]["final_reward"]) /
                         abs(results["no_curriculum"]["final_reward"]) * 100)
    
    results["comparison"] = {
        "reward_improvement_percent": reward_improvement,
    }
    
    return results


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark performance improvements")
    parser.add_argument("--config", default="configs/intersection.json", help="Environment config")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per benchmark")
    parser.add_argument("--out", default="runs/benchmarks", help="Output directory")
    parser.add_argument("--benchmark", choices=["per", "curriculum", "all"], default="all", help="Which benchmark to run")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    results = {}
    
    if args.benchmark in ["per", "all"]:
        print("=" * 60)
        print("PER vs Uniform Replay Benchmark")
        print("=" * 60)
        per_results = benchmark_per_vs_uniform(args.config, args.episodes)
        results["per_vs_uniform"] = per_results
        
        print(f"\nResults:")
        print(f"  Uniform Replay: {per_results['uniform']['final_reward']:.2f} ± {per_results['uniform']['std_reward']:.2f}")
        print(f"  PER:             {per_results['per']['final_reward']:.2f} ± {per_results['per']['std_reward']:.2f}")
        print(f"  Improvement:     {per_results['comparison']['reward_improvement_percent']:+.1f}%")
        print(f"  Time Overhead:   {per_results['comparison']['time_overhead_percent']:+.1f}%")
    
    if args.benchmark in ["curriculum", "all"]:
        print("\n" + "=" * 60)
        print("Curriculum Learning Benchmark")
        print("=" * 60)
        curriculum_results = benchmark_curriculum_learning(args.config, args.episodes)
        results["curriculum"] = curriculum_results
        
        print(f"\nResults:")
        print(f"  No Curriculum:   {curriculum_results['no_curriculum']['final_reward']:.2f} ± {curriculum_results['no_curriculum']['std_reward']:.2f}")
        print(f"  With Curriculum: {curriculum_results['curriculum']['final_reward']:.2f} ± {curriculum_results['curriculum']['std_reward']:.2f}")
        print(f"  Improvement:     {curriculum_results['comparison']['reward_improvement_percent']:+.1f}%")
        print(f"  Final Level:     {curriculum_results['curriculum']['final_level']}")
    
    # Save results
    results_path = os.path.join(args.out, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

