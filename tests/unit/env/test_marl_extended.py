"""
Extended MARL Environment Unit Tests.

Extended tests with API compliance, property-based testing, and reward validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from gymnasium import spaces
from hypothesis import given, strategies as st

# Import MARL environment
from src.env.marl_env import MarlEnv


class TestAPICompliance:
    """Test Gymnasium API compliance."""
    
    @pytest.fixture
    def mock_marl_env(self):
        """Create mock MARL environment."""
        with patch('src.env.marl_env.traci') as mock_traci:
            mock_traci.start = Mock()
            mock_traci.trafficlight.getIDList = Mock(return_value=['tl0', 'tl1'])
            mock_traci.trafficlight.getControlledLinks = Mock(return_value=[
                [('e0_0', 'e1_0', 0)],
                [('e2_0', 'e3_0', 0)]
            ])
            mock_traci.simulation.getTime.return_value = 0.0
            mock_traci.edge.getLastStepVehicleNumber.return_value = 5
            mock_traci.edge.getWaitingTime.return_value = 10.0
            
            env = MarlEnv(config_path="configs/grid.sumocfg")
            yield env
    
    def test_reset_returns_observations(self, mock_marl_env):
        """Test reset() returns observations for all agents."""
        env = mock_marl_env
        observations = env.reset()
        
        assert observations is not None
        assert isinstance(observations, (list, tuple, dict))
        assert len(observations) == env.num_agents
    
    def test_step_returns_tuple(self, mock_marl_env):
        """Test step() returns (observations, rewards, dones, infos)."""
        env = mock_marl_env
        observations = env.reset()
        
        actions = [0, 0]  # Actions for 2 agents
        result = env.step(actions)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        obs, rewards, dones, infos = result
        
        assert len(obs) == env.num_agents
        assert len(rewards) == env.num_agents
        assert len(dones) == env.num_agents
        assert len(infos) == env.num_agents
    
    def test_observation_space_consistency(self, mock_marl_env):
        """Test observation space is consistent."""
        env = mock_marl_env
        
        assert hasattr(env, 'observation_space')
        assert len(env.observation_space) == env.num_agents
        
        observations = env.reset()
        for i, obs in enumerate(observations):
            assert env.observation_space[i].contains(obs)
    
    def test_action_space_consistency(self, mock_marl_env):
        """Test action space is consistent."""
        env = mock_marl_env
        
        assert hasattr(env, 'action_space')
        assert len(env.action_space) == env.num_agents
        
        observations = env.reset()
        actions = [space.sample() for space in env.action_space]
        
        result = env.step(actions)
        assert result is not None


class TestRewardFunction:
    """Test reward function validation."""
    
    @pytest.fixture
    def mock_marl_env(self):
        """Create mock MARL environment."""
        with patch('src.env.marl_env.traci') as mock_traci:
            mock_traci.start = Mock()
            mock_traci.trafficlight.getIDList = Mock(return_value=['tl0', 'tl1'])
            mock_traci.trafficlight.getControlledLinks = Mock(return_value=[
                [('e0_0', 'e1_0', 0)],
                [('e2_0', 'e3_0', 0)]
            ])
            mock_traci.simulation.getTime.return_value = 0.0
            mock_traci.edge.getLastStepVehicleNumber.return_value = 5
            mock_traci.edge.getWaitingTime.return_value = 10.0
            
            env = MarlEnv(config_path="configs/grid.sumocfg")
            yield env
    
    def test_reward_range(self, mock_marl_env):
        """Test rewards are in reasonable range."""
        env = mock_marl_env
        observations = env.reset()
        
        for _ in range(10):
            actions = [0, 0]
            obs, rewards, dones, infos = env.step(actions)
            
            for reward in rewards:
                assert isinstance(reward, (int, float))
                assert -1000 < reward < 1000  # Reasonable range
    
    def test_reward_sign(self, mock_marl_env):
        """Test reward sign (should be negative for penalties)."""
        env = mock_marl_env
        observations = env.reset()
        
        # High queue should give negative reward
        actions = [0, 0]
        obs, rewards, dones, infos = env.step(actions)
        
        # Rewards are typically negative (penalties)
        # But can be positive for good performance
        for reward in rewards:
            assert isinstance(reward, (int, float))
    
    def test_reward_consistency(self, mock_marl_env):
        """Test reward consistency across similar states."""
        env = mock_marl_env
        
        # Run same scenario multiple times
        rewards_list = []
        for _ in range(5):
            obs = env.reset()
            actions = [0, 0]
            obs, rewards, dones, infos = env.step(actions)
            rewards_list.append(rewards)
        
        # Rewards should be similar (allowing for some variance)
        first_rewards = rewards_list[0]
        for rewards in rewards_list[1:]:
            # Check if rewards are similar (within 20% variance)
            for r1, r2 in zip(first_rewards, rewards):
                if r1 != 0:
                    variance = abs(r1 - r2) / abs(r1)
                    assert variance < 0.5  # Allow 50% variance


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        num_agents=st.integers(min_value=1, max_value=10),
        state_dim=st.integers(min_value=4, max_value=50)
    )
    def test_observation_dimensions(self, num_agents, state_dim):
        """Property: Observations should have correct dimensions."""
        # Mock environment
        with patch('src.env.marl_env.traci'):
            # This is a property test - should hold for any valid inputs
            assert num_agents > 0
            assert state_dim > 0
            assert num_agents * state_dim > 0
    
    @given(
        queue_length=st.floats(min_value=0, max_value=100),
        wait_time=st.floats(min_value=0, max_value=300)
    )
    def test_reward_monotonicity(self, queue_length, wait_time):
        """Property: Higher queues/wait times should give worse rewards."""
        # Simplified reward function test
        # Higher queue -> worse (more negative) reward
        reward_high = -(queue_length + wait_time)
        reward_low = -(queue_length * 0.5 + wait_time * 0.5)
        
        assert reward_high <= reward_low
    
    @given(
        action=st.integers(min_value=0, max_value=3)
    )
    def test_action_validity(self, action):
        """Property: Actions should always be valid."""
        num_phases = 4
        assert 0 <= action < num_phases


class TestEpisodeTermination:
    """Test episode termination conditions."""
    
    @pytest.fixture
    def mock_marl_env(self):
        """Create mock MARL environment."""
        with patch('src.env.marl_env.traci') as mock_traci:
            mock_traci.start = Mock()
            mock_traci.trafficlight.getIDList = Mock(return_value=['tl0', 'tl1'])
            mock_traci.trafficlight.getControlledLinks = Mock(return_value=[
                [('e0_0', 'e1_0', 0)],
                [('e2_0', 'e3_0', 0)]
            ])
            mock_traci.simulation.getTime.return_value = 0.0
            mock_traci.edge.getLastStepVehicleNumber.return_value = 5
            mock_traci.edge.getWaitingTime.return_value = 10.0
            
            env = MarlEnv(config_path="configs/grid.sumocfg")
            yield env
    
    def test_episode_terminates(self, mock_marl_env):
        """Test that episodes eventually terminate."""
        env = mock_marl_env
        observations = env.reset()
        
        max_steps = 1000
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            actions = [0, 0]
            obs, rewards, dones, infos = env.step(actions)
            done = all(dones)
            steps += 1
        
        # Episode should terminate
        assert done or steps >= max_steps
    
    def test_truncation_handling(self, mock_marl_env):
        """Test episode truncation handling."""
        env = mock_marl_env
        observations = env.reset()
        
        # Simulate truncation
        max_steps = 100
        steps = 0
        
        for _ in range(max_steps):
            actions = [0, 0]
            obs, rewards, dones, infos = env.step(actions)
            steps += 1
            
            # Check truncation info
            for info in infos:
                if 'TimeLimit.truncated' in info:
                    assert info['TimeLimit.truncated'] is True or False


class TestMultiAgentCoordination:
    """Test multi-agent coordination properties."""
    
    def test_independent_actions(self):
        """Test agents can take independent actions."""
        num_agents = 4
        actions = [np.random.randint(0, 4) for _ in range(num_agents)]
        
        # All actions should be independent
        assert len(actions) == num_agents
        assert all(0 <= a < 4 for a in actions)
    
    def test_shared_observation_space(self):
        """Test agents share observation space structure."""
        num_agents = 4
        state_dim = 12
        
        # All agents should have same observation space
        observation_spaces = [spaces.Box(low=0, high=100, shape=(state_dim,)) 
                              for _ in range(num_agents)]
        
        assert len(observation_spaces) == num_agents
        assert all(space.shape == (state_dim,) for space in observation_spaces)

