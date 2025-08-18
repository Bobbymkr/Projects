import argparse
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig

def analyze_dqn_decisions(cfg_path: str, model_path: str, steps: int = 400, seed: int = 42):
    """Analyze DQN agent's decision-making patterns"""
    # Load environment and agent
    cfg = load_config(cfg_path)
    env = TrafficEnv(cfg)
    
    agent = DQNAgent(state_dim=env.observation_space.shape[0], 
                    action_dim=env.action_space.n, 
                    cfg=DQNConfig())
    agent.load(model_path)
    
    obs, _ = env.reset(seed=seed)
    
    history: List[Dict[str, Any]] = []
    
    # Run simulation and collect detailed metrics
    for _ in range(steps):
        # Get Q-values for current state
        q_values = agent.get_q_values(obs.astype(np.float32))
        action = agent.select_action(obs.astype(np.float32), evaluate=True)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store detailed information
        history.append({
            "time": info.get("time", 0),
            "phase": info.get("phase", 0),
            "queues": next_obs.copy().tolist(),
            "reward": float(reward),
            "action_index": int(action),
            "green_time": int(env.green_values[action]),
            "q_values": q_values.tolist(),
            "state": obs.copy().tolist()
        })
        
        obs = next_obs
        if terminated or truncated:
            break
    
    # Analyze patterns
    phases = np.array([h["phase"] for h in history])
    queues = np.array([h["queues"] for h in history])
    actions = np.array([h["green_time"] for h in history])
    q_values = np.array([h["q_values"] for h in history])
    
    # Calculate key metrics
    avg_green_by_phase = {0: [], 1: []}
    avg_queue_by_phase = {0: [], 1: []}
    
    for phase in [0, 1]:
        phase_mask = phases == phase
        avg_green_by_phase[phase] = np.mean(actions[phase_mask]) if any(phase_mask) else 0
        avg_queue_by_phase[phase] = np.mean(queues[phase_mask], axis=0) if any(phase_mask) else np.zeros(4)
    
    # Print analysis
    print(f"\nAnalysis for {cfg_path}:")
    print("\nTraffic Patterns:")
    print(f"Arrival rates: {cfg['arrival_rates']}")
    
    print("\nAgent Decision Patterns:")
    for phase in [0, 1]:
        print(f"\nPhase {phase}:")
        print(f"  Average green time: {avg_green_by_phase[phase]:.1f}s")
        print(f"  Average queue lengths: {avg_queue_by_phase[phase]}")
    
    print("\nDecision Quality Metrics:")
    print(f"Q-value variance: {np.var(q_values):.2f}")
    print(f"Action entropy: {calculate_action_entropy(actions):.2f}")
    
    # Analyze state-action relationships
    print("\nState-Action Correlation:")
    for lane in range(4):
        correlation = np.corrcoef(queues[:, lane], actions)[0, 1]
        print(f"Lane {lane} queue vs action correlation: {correlation:.2f}")

def calculate_action_entropy(actions):
    """Calculate action distribution entropy as a measure of decision diversity"""
    unique_actions, counts = np.unique(actions, return_counts=True)
    probabilities = counts / len(actions)
    return -np.sum(probabilities * np.log(probabilities))

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
