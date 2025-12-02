import argparse
import json
from typing import Dict, Any, List
import numpy as np

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig

def analyze_dqn_decisions(cfg_path: str, model_path: str, steps: int = 400):
    """Analyze DQN agent's decision-making patterns"""
    # Load environment and agent
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)
    
    agent = DQNAgent(state_dim=env.observation_space.shape[0], 
                    action_dim=env.action_space.n, 
                    cfg=DQNConfig())
    agent.load(model_path)
    
    obs, _ = env.reset()
    
    history: List[Dict[str, Any]] = []
    
    # Run simulation and collect detailed metrics
    for _ in range(steps):
        action = agent.select_action(obs.astype(np.float32), evaluate=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store detailed information
        history.append({
            "time": info.get("time", 0),
            "phase": info.get("phase", 0),
            "queues": next_obs.copy().tolist(),
            "reward": float(reward),
            "action_index": int(action),
            "green_time": int(env.green_values[action])
        })
        
        obs = next_obs
        if terminated or truncated:
            break
    
    # Convert to numpy arrays for analysis
    phases = np.array([h["phase"] for h in history])
    queues = np.array([h["queues"] for h in history])
    actions = np.array([h["green_time"] for h in history])
    rewards = np.array([h["reward"] for h in history])
    
    # Analyze action patterns per phase
    phase_0_mask = phases == 0
    phase_1_mask = phases == 1
    
    print(f"\nAnalysis for {cfg_path}:")
    print("\nTraffic Patterns:")
    print(f"Arrival rates: {cfg['arrival_rates']}")
    
    print("\nPhase 0 (North-South):")
    if any(phase_0_mask):
        print(f"Average green time: {np.mean(actions[phase_0_mask]):.1f}s")
        print(f"Average queue lengths: {np.mean(queues[phase_0_mask], axis=0)}")
        print(f"Queue variance: {np.var(queues[phase_0_mask], axis=0)}")
    
    print("\nPhase 1 (East-West):")
    if any(phase_1_mask):
        print(f"Average green time: {np.mean(actions[phase_1_mask]):.1f}s")
        print(f"Average queue lengths: {np.mean(queues[phase_1_mask], axis=0)}")
        print(f"Queue variance: {np.var(queues[phase_1_mask], axis=0)}")
    
    print("\nDecision Analysis:")
    print(f"Action distribution: {np.bincount(actions.astype(int))}")
    print(f"Average reward per step: {np.mean(rewards):.2f}")
    print(f"Reward variance: {np.var(rewards):.2f}")
    
    # Analyze state-action relationships
    print("\nState-Action Correlation:")
    for lane in range(4):
        correlation = np.corrcoef(queues[:, lane], actions)[0, 1]
        print(f"Lane {lane} queue vs action correlation: {correlation:.2f}")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    scenarios = [
        ("configs/morning_rush.json", "Morning Rush Hour"),
        ("configs/intersection.json", "Mid-day Normal"),
        ("configs/evening_rush.json", "Evening Rush Hour")
    ]
    
    for cfg_path, scenario_name in scenarios:
        print(f"\n{'-'*20} {scenario_name} {'-'*20}")
        analyze_dqn_decisions(cfg_path, "runs/dqn_traffic.npz")
