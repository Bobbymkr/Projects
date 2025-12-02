"""
Unit tests for traffic environment.
"""

import pytest
import numpy as np
from src.env.traffic_env import TrafficEnv


class TestTrafficEnv:
    """Test traffic environment."""
    
    @pytest.fixture
    def env(self):
        """Create traffic environment."""
        config = {
            "num_lanes": 4,
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "arrival_rates": [0.3, 0.25, 0.35, 0.2]
        }
        return TrafficEnv(config)
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == env.num_lanes
        assert isinstance(info, dict)
    
    def test_step(self, env):
        """Test environment step."""
        obs, info = env.reset()
        action = 0
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_action_space(self, env):
        """Test action space."""
        assert env.action_space is not None
        assert env.action_space.n > 0
    
    def test_observation_space(self, env):
        """Test observation space."""
        assert env.observation_space is not None
        assert env.observation_space.shape == (env.num_lanes,)

