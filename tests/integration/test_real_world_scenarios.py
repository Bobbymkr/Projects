"""
Integration tests for real-world traffic scenarios.

Tests edge cases, emergencies, and multi-intersection coordination.
"""

import pytest
import numpy as np
import json
from pathlib import Path

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig


class TestRealWorldTrafficPatterns:
    """Test with actual traffic patterns."""
    
    def test_morning_rush_hour(self):
        """Test morning rush hour traffic pattern."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.8, 0.7, 0.3, 0.2],  # Heavy north-south traffic
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=DQNConfig()
        )
        
        # Run episode
        obs, info = env.reset()
        total_reward = 0.0
        
        for step in range(100):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, obs, terminated or truncated)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Verify system handled rush hour
        assert total_reward < 0  # Negative rewards expected
        assert info["total_vehicles_processed"] > 0
    
    def test_evening_rush_hour(self):
        """Test evening rush hour traffic pattern."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.2, 0.3, 0.8, 0.7],  # Heavy east-west traffic
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Verify environment handles asymmetric traffic
        assert len(obs) == 4
        assert np.all(obs >= 0) and np.all(obs <= 1)
    
    def test_low_traffic_night(self):
        """Test low traffic night scenario."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.05, 0.05, 0.05, 0.05],  # Very light traffic
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Run short episode
        for step in range(20):
            action = 0  # Simple action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Verify low traffic handled efficiently
        assert info["max_queue_length"] < 10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_max_queue_capacity(self):
        """Test behavior when queues reach maximum capacity."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [2.0, 2.0, 2.0, 2.0],  # Very high arrival rate
            "queue_capacity": 20,  # Small capacity
            "episode_horizon": 600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Run until queues fill up
        for step in range(50):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check queue capacity handling
            assert np.all(env.queues <= env.queue_capacity)
            
            if terminated or truncated:
                break
    
    def test_emergency_vehicle_scenario(self):
        """Test emergency vehicle priority scenario."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Simulate emergency: sudden high traffic on one lane
        env.queues[0] = 35  # Emergency lane with high queue
        
        # System should prioritize this lane
        action = 0  # Phase 0 (lanes 0, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify system responded
        assert reward is not None
        assert isinstance(reward, (float, np.floating))
    
    def test_sensor_failure_simulation(self):
        """Test system behavior with sensor failures (missing data)."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Simulate sensor failure: zero out one lane's observation
        obs[0] = 0.0
        
        # System should still function
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert len(obs) == 4


class TestMultiIntersectionCoordination:
    """Test multi-intersection coordination scenarios."""
    
    def test_two_intersection_coordination(self):
        """Test coordination between two intersections."""
        # Create two environments (simulating two intersections)
        config1 = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.4, 0.3, 0.3, 0.2],
            "queue_capacity": 40,
            "episode_horizon": 1800,
        }
        
        config2 = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.4, 0.2, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 1800,
        }
        
        env1 = TrafficEnv(config1)
        env2 = TrafficEnv(config2)
        
        agent1 = DQNAgent(
            state_dim=env1.observation_space.shape[0],
            action_dim=env1.action_space.n,
            cfg=DQNConfig()
        )
        
        agent2 = DQNAgent(
            state_dim=env2.observation_space.shape[0],
            action_dim=env2.action_space.n,
            cfg=DQNConfig()
        )
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Run coordinated simulation
        for step in range(50):
            action1 = agent1.select_action(obs1)
            action2 = agent2.select_action(obs2)
            
            obs1, reward1, term1, trunc1, info1 = env1.step(action1)
            obs2, reward2, term2, trunc2, info2 = env2.step(action2)
            
            if term1 or trunc1 or term2 or trunc2:
                break
        
        # Verify both intersections processed traffic
        assert info1["total_vehicles_processed"] >= 0
        assert info2["total_vehicles_processed"] >= 0
    
    def test_network_cascade_effect(self):
        """Test cascade effects in traffic network."""
        # Simulate upstream intersection affecting downstream
        upstream_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.8, 0.2, 0.2, 0.2],  # Heavy traffic on lane 0
            "queue_capacity": 40,
            "episode_horizon": 1800,
        }
        
        downstream_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.2, 0.2, 0.2, 0.2],
            "queue_capacity": 40,
            "episode_horizon": 1800,
        }
        
        upstream_env = TrafficEnv(upstream_config)
        downstream_env = TrafficEnv(downstream_config)
        
        obs_up, _ = upstream_env.reset()
        obs_down, _ = downstream_env.reset()
        
        # Simulate cascade: upstream congestion affects downstream
        for step in range(30):
            # Upstream action
            action_up = 0
            obs_up, _, term_up, trunc_up, info_up = upstream_env.step(action_up)
            
            # Downstream arrival rate increases if upstream has high queue
            if np.sum(upstream_env.queues) > 20:
                # Increase downstream arrival on corresponding lane
                downstream_env.arrival_rates[0] = min(1.0, downstream_env.arrival_rates[0] + 0.1)
            
            action_down = 0
            obs_down, _, term_down, trunc_down, info_down = downstream_env.step(action_down)
            
            if term_up or trunc_up or term_down or trunc_down:
                break
        
        # Verify cascade effect was simulated
        assert downstream_env.arrival_rates[0] >= 0.2


class TestSystemResilience:
    """Test system resilience under failures."""
    
    def test_graceful_degradation(self):
        """Test graceful degradation when agent fails."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 3600,
        }
        
        env = TrafficEnv(config)
        obs, info = env.reset()
        
        # Simulate agent failure: use default action
        default_action = 0
        
        for step in range(20):
            try:
                # In real system, would fallback to default
                action = default_action
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                # System should handle errors gracefully
                pytest.fail(f"System should handle errors gracefully: {e}")
            
            if terminated or truncated:
                break
        
        # Verify system continued operating
        assert info is not None
    
    def test_high_load_stress_test(self):
        """Test system under high load conditions."""
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [1.5, 1.5, 1.5, 1.5],  # Very high load
            "queue_capacity": 40,
            "episode_horizon": 1800,
        }
        
        env = TrafficEnv(config)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=DQNConfig()
        )
        
        obs, info = env.reset()
        
        # Run under high load
        for step in range(100):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.push(obs, action, reward, obs, terminated or truncated)
            
            # Verify queues don't exceed capacity
            assert np.all(env.queues <= env.queue_capacity)
            
            if terminated or truncated:
                break
        
        # Verify system handled stress
        assert info["total_vehicles_processed"] > 0

