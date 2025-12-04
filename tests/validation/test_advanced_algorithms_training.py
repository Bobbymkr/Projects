"""
Validation Tests for Advanced Algorithm Training.

Ensures HRL and MBRL can be trained and produce valid results.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.traffic_env import TrafficEnv
from src.research.novel_algorithms.hierarchical_rl_complete import CompleteHierarchicalRLAgent
from src.research.novel_algorithms.model_based_rl_complete import CompleteModelBasedRLAgent


class TestHRLLTraining:
    """Test HRL training and validation."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = {"num_lanes": 4}
        return TrafficEnv(config)
    
    @pytest.fixture
    def agent(self):
        """Create HRL agent."""
        return CompleteHierarchicalRLAgent(
            state_dim=12,
            action_dim=4,
            use_domain_options=True,
        )
    
    def test_hrl_can_be_trained(self, agent, env):
        """Test that HRL agent can be trained."""
        # Collect some experience
        for episode in range(10):
            state = env.reset()
            trajectory = []
            
            for step in range(20):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, next_state))
                state = next_state
                if done:
                    break
            
            agent.option_trajectories.append(trajectory)
            agent.primitive_trajectories.append(trajectory)
        
        # Train agent
        train_stats = agent.train(episodes=1, batch_size=16)
        
        assert train_stats is not None
        assert "episode_rewards" in train_stats
    
    def test_hrl_produces_valid_actions(self, agent, env):
        """Test that HRL produces valid actions."""
        state = env.reset()
        
        for _ in range(10):
            action = agent.select_action(state)
            assert 0 <= action < 4
            state, _, _, _ = env.step(action)
    
    def test_hrl_options_are_available(self, agent):
        """Test that HRL has options available."""
        assert len(agent.options) > 0
        assert agent.hierarchical_policy is not None


class TestMBRLTraining:
    """Test MBRL training and validation."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = {"num_lanes": 4}
        return TrafficEnv(config)
    
    @pytest.fixture
    def agent(self):
        """Create MBRL agent."""
        return CompleteModelBasedRLAgent(
            state_dim=12,
            action_dim=4,
        )
    
    def test_mbrl_can_be_trained(self, agent, env):
        """Test that MBRL agent can be trained."""
        # Collect transitions
        for episode in range(10):
            state = env.reset()
            for step in range(20):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.add_transition(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
        
        # Train world model
        if len(agent.transition_buffer) >= 32:
            model_stats = agent.train_world_model(epochs=10, batch_size=32)
            assert model_stats is not None
            assert agent.world_model.is_trained
    
    def test_mbrl_produces_valid_actions(self, agent, env):
        """Test that MBRL produces valid actions."""
        state = env.reset()
        
        for _ in range(10):
            action = agent.select_action(state)
            assert 0 <= action < 4
            state, _, _, _ = env.step(action)
    
    def test_mbrl_world_model_trains(self, agent, env):
        """Test that world model can be trained."""
        # Collect transitions
        for _ in range(50):
            state = env.reset()
            for step in range(10):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.add_transition(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
        
        # Train
        if len(agent.transition_buffer) >= 32:
            stats = agent.train_world_model(epochs=20, batch_size=32)
            assert agent.world_model.is_trained
            assert "transition_loss" in stats or "loss" in stats


class TestAlgorithmComparison:
    """Test algorithm comparison capabilities."""
    
    def test_algorithms_can_be_compared(self):
        """Test that algorithms can be compared."""
        from src.control.fuzzy_control import FuzzyController
        
        # Create agents
        hrl_agent = CompleteHierarchicalRLAgent(state_dim=12, action_dim=4)
        mbrl_agent = CompleteModelBasedRLAgent(state_dim=12, action_dim=4)
        fuzzy_controller = FuzzyController()
        
        # All should be instantiable
        assert hrl_agent is not None
        assert mbrl_agent is not None
        assert fuzzy_controller is not None
        
        # All should be able to make decisions
        state = np.random.rand(12)
        
        hrl_action = hrl_agent.select_action(state)
        mbrl_action = mbrl_agent.select_action(state)
        fuzzy_action = fuzzy_controller.compute_timing(state[:4].tolist(), state[4:8].tolist())
        
        assert 0 <= hrl_action < 4
        assert 0 <= mbrl_action < 4
        assert fuzzy_action is not None

