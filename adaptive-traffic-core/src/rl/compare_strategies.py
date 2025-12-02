import argparse
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def simulate_fixed_timing(cfg_path: str, fixed_green: int, steps: int, seed: int) -> Dict[str, Any]:
    """Run simulation with fixed timing strategy"""
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)
    obs, _ = env.reset(seed=seed)
    
    history: List[Dict[str, Any]] = []
    # Always choose the action that gives the fixed green time
    fixed_action = (fixed_green - env.min_green) // env.green_step
    
    for _ in range(steps):
        next_obs, reward, terminated, truncated, info = env.step(fixed_action)
        history.append({
            "time": info.get("time", 0),
            "phase": info.get("phase", 0),
            "queues": next_obs.copy().tolist(),
            "reward": float(reward),
            "action_index": int(fixed_action),
            "green": fixed_green,
        })
        obs = next_obs
        if terminated or truncated:
            break
            
    return {"env": env, "history": history}

def simulate_dqn(cfg_path: str, model_path: str, steps: int, seed: int) -> Dict[str, Any]:
    """Run simulation with DQN agent"""
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)
    
    agent = DQNAgent(state_dim=env.observation_space.shape[0], 
                    action_dim=env.action_space.n, 
                    cfg=DQNConfig())
    agent.load(model_path)
    
    obs, _ = env.reset(seed=seed)
    history: List[Dict[str, Any]] = []
    
    for _ in range(steps):
        action = agent.select_action(obs.astype(np.float32), evaluate=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        history.append({
            "time": info.get("time", 0),
            "phase": info.get("phase", 0),
            "queues": next_obs.copy().tolist(),
            "reward": float(reward),
            "action_index": int(action),
            "green": int(env.green_values[action]),
        })
        obs = next_obs
        if terminated or truncated:
            break
            
    return {"env": env, "history": history}

def run_comparison(cfg_path: str, model_path: str, steps: int = 400, seed: int = 42):
    # Run DQN simulation
    dqn_results = simulate_dqn(cfg_path, model_path, steps, seed)
    
    # Run fixed timing simulations with different cycle lengths
    fixed_results = {}
    for fixed_green in [30, 45, 60]:  # Try different fixed timings
        fixed_results[fixed_green] = simulate_fixed_timing(cfg_path, fixed_green, steps, seed)
    
    # Calculate metrics
    dqn_rewards = [h["reward"] for h in dqn_results["history"]]
    dqn_queues = [np.mean(h["queues"]) for h in dqn_results["history"]]
    
    fixed_metrics = {}
    for green_time, result in fixed_results.items():
        rewards = [h["reward"] for h in result["history"]]
        queues = [np.mean(h["queues"]) for h in result["history"]]
        fixed_metrics[green_time] = {
            "avg_reward": np.mean(rewards),
            "avg_queue": np.mean(queues),
            "max_queue": np.max(queues)
        }
    
    # Print comparison
    print(f"\nResults for {cfg_path}:")
    print(f"DQN Agent:")
    print(f"  Average reward: {np.mean(dqn_rewards):.2f}")
    print(f"  Average queue length: {np.mean(dqn_queues):.2f}")
    print(f"  Maximum queue length: {np.max(dqn_queues):.2f}")
    
    print("\nFixed Timing Results:")
    for green_time, metrics in fixed_metrics.items():
        print(f"\nFixed {green_time}s green time:")
        print(f"  Average reward: {metrics['avg_reward']:.2f}")
        print(f"  Average queue length: {metrics['avg_queue']:.2f}")
        print(f"  Maximum queue length: {metrics['max_queue']:.2f}")

if __name__ == "__main__":
    # Test scenarios
    scenarios = [
        ("configs/morning_rush.json", "Morning Rush Hour"),
        ("configs/intersection.json", "Mid-day Normal"),
        ("configs/evening_rush.json", "Evening Rush Hour")
    ]
    
    for cfg_path, scenario_name in scenarios:
        print(f"\n{'-'*20} {scenario_name} {'-'*20}")
        run_comparison(cfg_path, "runs/dqn_traffic.npz")
