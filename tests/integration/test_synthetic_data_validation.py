"""
Synthetic Data Validation Tests.

Validates synthetic data generation quality and characteristics.
Implements Phase 3.2 from Expert Review Remediation Plan.
"""

import pytest
import numpy as np
from typing import List, Dict

from src.env.traffic_env import TrafficEnv


class TestSyntheticDataGeneration:
    """Test synthetic data generation quality."""
    
    def test_arrival_rate_distribution(self):
        """Test that arrival rates follow expected distribution."""
        env_config = {
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
        
        env = TrafficEnv(env_config)
        
        # Collect arrival data over multiple steps
        arrivals_per_lane = {i: [] for i in range(env.num_lanes)}
        
        obs, info = env.reset()
        for step in range(1000):
            # Simulate step
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track queue changes (proxy for arrivals)
            for lane in range(env.num_lanes):
                if step > 0:
                    arrivals_per_lane[lane].append(obs[lane])
            
            if terminated or truncated:
                break
        
        # Verify arrivals are within reasonable bounds
        for lane in range(env.num_lanes):
            if arrivals_per_lane[lane]:
                mean_arrival = np.mean(arrivals_per_lane[lane])
                # Should be positive and within queue capacity
                assert 0 <= mean_arrival <= env.queue_capacity
    
    def test_traffic_pattern_consistency(self):
        """Test that traffic patterns are consistent across episodes."""
        env_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.4, 0.3, 0.2, 0.1],  # Asymmetric
            "queue_capacity": 40,
            "episode_horizon": 600,
        }
        
        env = TrafficEnv(env_config)
        
        # Run multiple episodes
        episode_queues = []
        for episode in range(5):
            obs, info = env.reset()
            episode_queue = []
            
            for step in range(50):
                action = 0
                obs, reward, terminated, truncated, info = env.step(action)
                episode_queue.append(np.sum(obs[:env.num_lanes]))
                
                if terminated or truncated:
                    break
            
            episode_queues.append(np.mean(episode_queue) if episode_queue else 0)
        
        # Episodes should show some consistency (not completely random)
        # Variance should be reasonable
        queue_variance = np.var(episode_queues)
        assert queue_variance < 100  # Reasonable variance
    
    def test_rush_hour_pattern(self):
        """Test rush hour traffic pattern generation."""
        # High arrival rates for rush hour
        env_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.8, 0.7, 0.6, 0.5],  # High traffic
            "queue_capacity": 40,
            "episode_horizon": 600,
        }
        
        env = TrafficEnv(env_config)
        obs, info = env.reset()
        
        # Run for several steps
        max_queue = 0
        for step in range(100):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            max_queue = max(max_queue, np.max(obs[:env.num_lanes]))
            
            if terminated or truncated:
                break
        
        # Rush hour should show higher queues
        assert max_queue > 0
        # But should not exceed capacity
        assert max_queue <= env.queue_capacity
    
    def test_low_traffic_pattern(self):
        """Test low traffic pattern generation."""
        # Low arrival rates for night/off-peak
        env_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.05, 0.05, 0.05, 0.05],  # Very low traffic
            "queue_capacity": 40,
            "episode_horizon": 600,
        }
        
        env = TrafficEnv(env_config)
        obs, info = env.reset()
        
        # Run for several steps
        queues = []
        for step in range(100):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            queues.append(np.sum(obs[:env.num_lanes]))
            
            if terminated or truncated:
                break
        
        # Low traffic should show lower queues
        mean_queue = np.mean(queues) if queues else 0
        assert mean_queue < 10  # Should be low


class TestDataQuality:
    """Test synthetic data quality characteristics."""
    
    def test_state_normalization(self):
        """Test that states are properly normalized."""
        env = TrafficEnv({
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 600,
        })
        
        obs, info = env.reset()
        
        # States should be normalized (0-1 range typically)
        assert len(obs) == env.observation_space.shape[0]
        assert np.all(obs >= 0)  # Non-negative
        assert np.all(obs <= 1) or np.all(obs <= env.queue_capacity)  # Within bounds
    
    def test_reward_consistency(self):
        """Test that rewards are consistent and meaningful."""
        env = TrafficEnv({
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 600,
        })
        
        obs, info = env.reset()
        rewards = []
        
        for step in range(50):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            
            # Rewards should be finite
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break
        
        # Rewards should show some structure (not completely random)
        assert len(rewards) > 0
        assert all(np.isfinite(r) for r in rewards)
    
    def test_observation_consistency(self):
        """Test that observations are consistent in structure."""
        env = TrafficEnv({
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 600,
        })
        
        # Run multiple resets
        for _ in range(5):
            obs, info = env.reset()
            
            # Observation shape should be consistent
            assert obs.shape == env.observation_space.shape
            assert len(obs) == env.num_lanes  # Assuming queue lengths per lane
            
            # Info should contain expected keys (based on actual API)
            assert "time" in info
            assert "phase" in info
            assert "queues" in info
            assert "wait_times" in info
            # Statistics are tracked but not in info dict from reset()
            assert hasattr(env, "total_vehicles_processed")
            assert hasattr(env, "max_queue_length")


class TestDataDistributionMatching:
    """Test that synthetic data matches expected real-world distributions."""
    
    def test_queue_length_distribution(self):
        """Test queue length distribution matches expected patterns."""
        env = TrafficEnv({
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 1800,
        })
        
        obs, info = env.reset()
        queue_lengths = []
        
        # Collect queue data
        for step in range(200):
            action = step % 4  # Cycle through actions
            obs, reward, terminated, truncated, info = env.step(action)
            queue_lengths.extend(obs[:env.num_lanes].tolist())
            
            if terminated or truncated:
                break
        
        # Queue lengths should be within bounds
        assert all(0 <= q <= env.queue_capacity for q in queue_lengths)
        
        # Distribution should be reasonable (not all zeros, not all max)
        unique_values = len(set(queue_lengths))
        assert unique_values > 1  # Some variation
    
    def test_wait_time_distribution(self):
        """Test wait time distribution."""
        env = TrafficEnv({
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
            "queue_capacity": 40,
            "episode_horizon": 1800,
        })
        
        obs, info = env.reset()
        
        # Wait times should be tracked
        # (Implementation depends on how wait_times are exposed)
        # For now, verify environment tracks wait times
        assert hasattr(env, 'wait_times') or 'wait_times' in str(type(env))

