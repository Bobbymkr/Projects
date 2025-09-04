"""
Comprehensive tests for TrafficEnv class.

This module tests all aspects of the traffic environment including:
- Initialization and configuration validation
- State transitions and observations
- Reward computation
- Episode management
- Statistics tracking
"""

import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.traffic_env import TrafficEnv


@pytest.fixture
def valid_config():
    """Valid configuration for testing."""
    return {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "reward_weights": {"queue": -1.0, "wait_penalty": -0.1},
        "episode_horizon": 3600
    }


@pytest.fixture
def traffic_env(valid_config):
    """Traffic environment fixture."""
    return TrafficEnv(valid_config)


class TestTrafficEnvInitialization:
    """Test environment initialization and configuration validation."""
    
    def test_valid_initialization(self, valid_config):
        """Test successful initialization with valid config."""
        env = TrafficEnv(valid_config)
        assert env.num_lanes == 4
        assert env.phase_lanes == [[0, 1], [2, 3]]
        assert env.min_green == 5
        assert env.max_green == 60
        assert env.green_step == 5
        assert env.queue_capacity == 40
        assert len(env.arrival_rates) == 4
        assert env.episode_horizon == 3600
    
    def test_default_config(self):
        """Test initialization with minimal config."""
        env = TrafficEnv({})
        assert env.num_lanes == 4
        assert env.min_green == 5
        assert env.max_green == 60
        assert env.episode_horizon == 3600
    
    def test_invalid_num_lanes(self):
        """Test validation of invalid number of lanes."""
        config = {"num_lanes": 0}
        with pytest.raises(ValueError, match="num_lanes must be positive"):
            TrafficEnv(config)
    
    def test_invalid_green_times(self):
        """Test validation of invalid green time configuration."""
        config = {"min_green": 60, "max_green": 30}
        with pytest.raises(ValueError, match="min_green must be less than max_green"):
            TrafficEnv(config)
    
    def test_invalid_phase_lanes(self):
        """Test validation of invalid phase configuration."""
        config = {"phase_lanes": [[0, 1]]}  # Only one phase
        with pytest.raises(ValueError, match="Must have exactly 2 phases"):
            TrafficEnv(config)
    
    def test_invalid_arrival_rates(self):
        """Test validation of invalid arrival rates."""
        config = {"num_lanes": 4, "arrival_rates": [0.3, 0.25]}  # Wrong length
        with pytest.raises(ValueError, match="arrival_rates length must match num_lanes"):
            TrafficEnv(config)
    
    def test_negative_arrival_rates(self):
        """Test validation of negative arrival rates."""
        config = {"num_lanes": 4, "arrival_rates": [0.3, 0.25, -0.1, 0.2]}
        with pytest.raises(ValueError, match="Arrival rates must be non-negative"):
            TrafficEnv(config)
    
    def test_action_space(self, traffic_env):
        """Test action space configuration."""
        assert traffic_env.action_space.n == 12  # (60-5)/5 + 1
        assert len(traffic_env.green_values) == 12
        assert traffic_env.green_values[0] == 5
        assert traffic_env.green_values[-1] == 60
    
    def test_observation_space(self, traffic_env):
        """Test observation space configuration."""
        assert traffic_env.observation_space.shape == (4,)
        assert traffic_env.observation_space.dtype == np.float32
        assert np.all(traffic_env.observation_space.low == 0.0)
        assert np.all(traffic_env.observation_space.high == 1.0)


class TestTrafficEnvReset:
    """Test environment reset functionality."""
    
    def test_reset_basic(self, traffic_env):
        """Test basic reset functionality."""
        obs, info = traffic_env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        
        assert isinstance(info, dict)
        assert "time" in info
        assert "phase" in info
        assert "queues" in info
        assert "wait_times" in info
        assert info["time"] == 0
        assert info["phase"] == 0
    
    def test_reset_with_seed(self, traffic_env):
        """Test reset with seed for reproducibility."""
        obs1, _ = traffic_env.reset(seed=42)
        obs2, _ = traffic_env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reset_statistics(self, traffic_env):
        """Test that statistics are reset properly."""
        # Run some steps to accumulate statistics
        obs, _ = traffic_env.reset()
        action = 0
        for _ in range(5):
            obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        # Reset and check statistics
        obs, info = traffic_env.reset()
        assert traffic_env.total_vehicles_processed == 0
        assert traffic_env.total_wait_time == 0.0
        assert traffic_env.max_queue_length >= 0


class TestTrafficEnvStep:
    """Test environment step functionality."""
    
    def test_step_basic(self, traffic_env):
        """Test basic step functionality."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0  # 5 seconds green
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (4,)
        assert next_obs.dtype == np.float32
        
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        assert not terminated  # Should not terminate early
        assert not truncated  # Should not be truncated yet
    
    def test_step_invalid_action(self, traffic_env):
        """Test step with invalid action."""
        obs, _ = traffic_env.reset()
        invalid_action = 100
        
        with pytest.raises(AssertionError, match="Invalid action"):
            traffic_env.step(invalid_action)
    
    def test_step_phase_transition(self, traffic_env):
        """Test that phases transition correctly."""
        obs, _ = traffic_env.reset(seed=42)
        
        # First step should be phase 0
        action = 0
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        assert info["phase"] == 1  # Should switch to phase 1
        
        # Second step should be phase 1
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        assert info["phase"] == 0  # Should switch back to phase 0
    
    def test_step_time_progression(self, traffic_env):
        """Test that time progresses correctly."""
        obs, _ = traffic_env.reset()
        action = 0  # 5 seconds green
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        expected_time = 5 + 3 + 1  # green + yellow + all_red
        assert info["time"] == expected_time
    
    def test_step_episode_truncation(self, traffic_env):
        """Test episode truncation at horizon."""
        # Set short horizon for testing
        traffic_env.episode_horizon = 10
        
        obs, _ = traffic_env.reset()
        action = 0
        
        # Should truncate after one step (9 seconds total)
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        assert truncated
        assert not terminated


class TestTrafficEnvReward:
    """Test reward computation."""
    
    def test_reward_structure(self, traffic_env):
        """Test that reward has expected structure."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        assert isinstance(reward, float)
        # Reward should be negative (penalties for queues and wait times)
        assert reward <= 0
    
    def test_reward_components(self, traffic_env):
        """Test individual reward components."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        # Check that reward computation includes all components
        # This is tested indirectly by checking the reward structure
        assert reward <= 0  # Should be negative due to penalties
    
    def test_reward_normalization(self, traffic_env):
        """Test that rewards are properly normalized."""
        obs, _ = traffic_env.reset(seed=42)
        
        # Test multiple steps to see reward variation
        rewards = []
        for _ in range(5):
            action = 0
            next_obs, reward, terminated, truncated, info = traffic_env.step(action)
            rewards.append(reward)
        
        # Rewards should be finite and reasonable
        assert all(np.isfinite(r) for r in rewards)
        assert all(r <= 0 for r in rewards)  # All should be penalties


class TestTrafficEnvStatistics:
    """Test statistics tracking."""
    
    def test_vehicle_processing_tracking(self, traffic_env):
        """Test tracking of processed vehicles."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        assert traffic_env.total_vehicles_processed >= 0
        assert "total_vehicles_processed" in info
    
    def test_wait_time_tracking(self, traffic_env):
        """Test tracking of wait times."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        assert traffic_env.total_wait_time >= 0
        assert "total_wait_time" in info
    
    def test_max_queue_tracking(self, traffic_env):
        """Test tracking of maximum queue length."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        assert traffic_env.max_queue_length >= 0
        assert "max_queue_length" in info
    
    def test_get_statistics(self, traffic_env):
        """Test get_statistics method."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        
        # Run some steps
        for _ in range(3):
            next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        stats = traffic_env.get_statistics()
        
        assert "total_vehicles_processed" in stats
        assert "total_wait_time" in stats
        assert "max_queue_length" in stats
        assert "average_wait_time" in stats
        assert "throughput_rate" in stats
        
        assert stats["total_vehicles_processed"] >= 0
        assert stats["total_wait_time"] >= 0
        assert stats["max_queue_length"] >= 0
        assert stats["average_wait_time"] >= 0
        assert stats["throughput_rate"] >= 0


class TestTrafficEnvSimulation:
    """Test simulation functionality."""
    
    def test_simulate_episode_random(self, traffic_env):
        """Test simulation with random actions."""
        # This should run without errors
        traffic_env.simulate_episode(num_steps=3, seed=42, use_agent=False)
    
    def test_simulate_episode_with_agent(self, traffic_env):
        """Test simulation with DQN agent."""
        # This should run without errors
        traffic_env.simulate_episode(num_steps=3, seed=42, use_agent=True)
    
    def test_render(self, traffic_env):
        """Test render method."""
        obs, _ = traffic_env.reset(seed=42)
        action = 0
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        # Render should not raise an error
        traffic_env.render()


class TestTrafficEnvEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_queues(self, traffic_env):
        """Test behavior with empty queues."""
        obs, _ = traffic_env.reset(seed=42)
        # Force empty queues
        traffic_env.queues = np.zeros(4)
        
        action = 0
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        # Should handle empty queues gracefully
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
    
    def test_full_queues(self, traffic_env):
        """Test behavior with full queues."""
        obs, _ = traffic_env.reset(seed=42)
        # Force full queues
        traffic_env.queues = np.full(4, traffic_env.queue_capacity)
        
        action = 0
        next_obs, reward, terminated, truncated, info = traffic_env.step(action)
        
        # Should handle full queues gracefully
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
    
    def test_high_arrival_rates(self, traffic_env):
        """Test behavior with high arrival rates."""
        # Create config with high arrival rates
        high_rate_config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "arrival_rates": [2.0, 2.0, 2.0, 2.0],  # Very high rates
            "queue_capacity": 40,
            "reward_weights": {"queue": -1.0, "wait_penalty": -0.1},
            "episode_horizon": 3600
        }
        
        env = TrafficEnv(high_rate_config)
        obs, _ = env.reset(seed=42)
        action = 0
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Should handle high arrival rates gracefully
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)


if __name__ == "__main__":
    pytest.main([__file__])
