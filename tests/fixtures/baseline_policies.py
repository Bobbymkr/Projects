"""
Baseline Policy Fixtures for Testing.

Provides fixed-time controllers, Webster's method outputs, fuzzy logic baselines,
and expert demonstrations for comparison testing.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Callable
from pathlib import Path
import json


@pytest.fixture
def fixed_time_baseline() -> Dict[str, Any]:
    """Fixed-time traffic signal controller baseline."""
    return {
        "cycle_length": 120,  # seconds
        "phase_durations": [30, 30, 30, 30],  # 4 phases, 30s each
        "yellow_time": 3,
        "all_red_time": 2,
        "offset": 0
    }


@pytest.fixture
def webster_method_baseline() -> Dict[str, Any]:
    """Webster's method baseline output."""
    return {
        "optimal_cycle_length": 90,
        "phase_durations": [25, 20, 25, 20],
        "yellow_time": 3,
        "all_red_time": 2,
        "lost_time": 4,
        "saturation_flow": [1800, 1800, 1800, 1800],  # vehicles/hour
        "flow_ratios": [0.4, 0.35, 0.45, 0.38]
    }


@pytest.fixture
def fuzzy_logic_baseline() -> Callable:
    """Fuzzy logic controller baseline function."""
    def fuzzy_control(queue_lengths: List[float], wait_times: List[float]) -> int:
        """Simple fuzzy logic baseline."""
        # Simple rule: prioritize lane with highest queue
        max_queue_idx = np.argmax(queue_lengths)
        
        # If wait time is high, extend green time
        if wait_times[max_queue_idx] > 30:
            return max_queue_idx
        
        # Otherwise, cycle through phases
        return max_queue_idx
    
    return fuzzy_control


@pytest.fixture
def expert_demonstrations() -> List[Dict[str, Any]]:
    """Expert demonstration trajectories for imitation learning."""
    demonstrations = []
    
    for episode in range(10):
        states = []
        actions = []
        rewards = []
        
        # Generate a trajectory
        for step in range(100):
            # State: queue lengths, wait times, current phase
            state = {
                "queue_lengths": np.random.randint(0, 20, 4).tolist(),
                "wait_times": np.random.uniform(0, 60, 4).tolist(),
                "current_phase": step % 4,
                "phase_duration": 30.0
            }
            
            # Expert action: choose phase with highest queue
            queue_lengths = np.array(state["queue_lengths"])
            action = int(np.argmax(queue_lengths))
            
            # Reward: negative of wait time
            reward = -np.mean(state["wait_times"])
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        demonstrations.append({
            "episode_id": episode,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_reward": sum(rewards)
        })
    
    return demonstrations


@pytest.fixture
def baseline_performance_metrics() -> Dict[str, Dict[str, float]]:
    """Expected performance metrics for baseline policies."""
    return {
        "fixed_time": {
            "avg_wait_time": 25.0,
            "max_wait_time": 60.0,
            "throughput": 800,  # vehicles/hour
            "queue_length": 8.5
        },
        "webster": {
            "avg_wait_time": 18.0,
            "max_wait_time": 45.0,
            "throughput": 950,
            "queue_length": 6.2
        },
        "fuzzy_logic": {
            "avg_wait_time": 8.51,  # Best baseline
            "max_wait_time": 30.0,
            "throughput": 1100,
            "queue_length": 4.1
        }
    }


@pytest.fixture
def optimal_policy_reference() -> Dict[str, Any]:
    """Reference optimal policy for comparison."""
    return {
        "avg_wait_time_target": 5.0,  # seconds
        "max_wait_time_target": 15.0,
        "throughput_target": 1500,  # vehicles/hour
        "queue_length_target": 2.0,
        "inference_latency_target": 0.01,  # seconds
        "convergence_episodes": 2000
    }

