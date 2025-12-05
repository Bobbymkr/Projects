"""
Unit tests for Model-Based RL Agent.

Tests for WorldModel, ModelPredictiveControl, and ModelBasedRLAgent.
"""

import pytest
import numpy as np
from typing import List

from src.research.novel_algorithms.model_based_rl import (
    WorldModel,
    ModelPredictiveControl,
    ModelBasedRLAgent,
    WorldModelState,
)


class TestWorldModel:
    """Test suite for WorldModel."""
    
    def test_world_model_initialization(self):
        """Test WorldModel initialization."""
        state_dim = 4
        action_dim = 12
        model = WorldModel(state_dim, action_dim)
        
        assert model.state_dim == state_dim
        assert model.action_dim == action_dim
        assert not model.is_trained
        assert model.transition_model is not None
        assert model.reward_model is not None
    
    def test_world_model_train(self):
        """Test WorldModel training."""
        model = WorldModel(state_dim=4, action_dim=12)
        
        # Create sample transitions
        transitions = [
            WorldModelState(
                state=np.array([0.1, 0.2, 0.3, 0.4]),
                action=0,
                next_state=np.array([0.2, 0.3, 0.4, 0.5]),
                reward=-1.0,
                done=False
            ) for _ in range(50)
        ]
        
        # Train model
        metrics = model.train(transitions, epochs=10, batch_size=32)
        
        assert isinstance(metrics, dict)
        assert model.is_trained
    
    def test_world_model_predict(self):
        """Test WorldModel prediction."""
        model = WorldModel(state_dim=4, action_dim=12)
        
        # Train first
        transitions = [
            WorldModelState(
                state=np.random.rand(4),
                action=np.random.randint(0, 12),
                next_state=np.random.rand(4),
                reward=np.random.randn(),
                done=False
            ) for _ in range(50)
        ]
        model.train(transitions, epochs=5, batch_size=32)
        
        # Test prediction using actual API methods
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0
        
        # Use actual API: predict_next_state and predict_reward
        next_state = model.predict_next_state(state, action)
        # predict_reward requires state, action, and next_state
        reward = model.predict_reward(state, action, next_state)
        
        assert next_state is not None
        assert next_state.shape == (4,)
        assert isinstance(reward, (float, np.floating))


class TestModelPredictiveControl:
    """Test suite for ModelPredictiveControl."""
    
    def test_mpc_initialization(self):
        """Test MPC initialization."""
        world_model = WorldModel(state_dim=4, action_dim=12)
        mpc = ModelPredictiveControl(world_model)
        
        assert mpc.world_model == world_model
        assert mpc.horizon > 0
        assert mpc.num_candidates > 0
    
    def test_mpc_select_action(self):
        """Test MPC action selection."""
        world_model = WorldModel(state_dim=4, action_dim=12)
        
        # Train world model
        transitions = [
            WorldModelState(
                state=np.random.rand(4),
                action=np.random.randint(0, 12),
                next_state=np.random.rand(4),
                reward=np.random.randn(),
                done=False
            ) for _ in range(50)
        ]
        world_model.train(transitions, epochs=5, batch_size=32)
        
        mpc = ModelPredictiveControl(world_model, horizon=5, num_candidates=10)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        action = mpc.select_action(state, action_dim=12)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 12
    
    def test_mpc_simulate_trajectory(self):
        """Test MPC trajectory simulation."""
        world_model = WorldModel(state_dim=4, action_dim=12)
        
        # Train world model
        transitions = [
            WorldModelState(
                state=np.random.rand(4),
                action=np.random.randint(0, 12),
                next_state=np.random.rand(4),
                reward=np.random.randn(),
                done=False
            ) for _ in range(50)
        ]
        world_model.train(transitions, epochs=5, batch_size=32)
        
        mpc = ModelPredictiveControl(world_model, horizon=5)
        initial_state = np.array([0.1, 0.2, 0.3, 0.4])
        action_sequence = [0, 1, 2, 3, 4]
        
        total_reward = mpc._simulate_trajectory(initial_state, action_sequence)
        
        assert isinstance(total_reward, (float, np.floating))


class TestModelBasedRLAgent:
    """Test suite for ModelBasedRLAgent."""
    
    def test_agent_initialization(self):
        """Test ModelBasedRLAgent initialization."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        assert agent.state_dim == 4
        assert agent.action_dim == 12
        assert agent.world_model is not None
        assert agent.mpc is not None
        assert len(agent.transition_buffer) == 0
    
    def test_agent_add_transition(self):
        """Test adding transitions to buffer."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0
        reward = -1.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False
        
        agent.add_transition(state, action, reward, next_state, done)
        
        assert len(agent.transition_buffer) == 1
        transition = agent.transition_buffer[0]
        assert np.array_equal(transition.state, state)
        assert transition.action == action
        assert transition.reward == reward
        assert np.array_equal(transition.next_state, next_state)
        assert transition.done == done
    
    def test_agent_train_world_model(self):
        """Test training world model."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        # Add transitions
        for i in range(50):
            agent.add_transition(
                state=np.random.rand(4),
                action=np.random.randint(0, 12),
                reward=np.random.randn(),
                next_state=np.random.rand(4),
                done=False
            )
        
        # Train world model
        metrics = agent.train_world_model(epochs=10, batch_size=32)
        
        assert isinstance(metrics, dict)
        assert agent.world_model.is_trained
    
    def test_agent_select_action(self):
        """Test action selection."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        # Train world model first
        for i in range(50):
            agent.add_transition(
                state=np.random.rand(4),
                action=np.random.randint(0, 12),
                reward=np.random.randn(),
                next_state=np.random.rand(4),
                done=False
            )
        agent.train_world_model(epochs=5, batch_size=32)
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = agent.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 12
    
    def test_agent_reset_buffer(self):
        """Test buffer reset."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        # Add some transitions
        for i in range(10):
            agent.add_transition(
                state=np.random.rand(4),
                action=0,
                reward=-1.0,
                next_state=np.random.rand(4),
                done=False
            )
        
        assert len(agent.transition_buffer) == 10
        
        agent.reset_buffer()
        
        assert len(agent.transition_buffer) == 0


class TestModelBasedRLIntegration:
    """Integration tests for Model-Based RL."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle."""
        agent = ModelBasedRLAgent(state_dim=4, action_dim=12)
        
        # Simulate training episodes
        for episode in range(5):
            state = np.random.rand(4)
            
            for step in range(10):
                # Select action
                action = agent.select_action(state)
                
                # Simulate environment step
                next_state = state + np.random.randn(4) * 0.1
                reward = -np.sum(np.abs(next_state))
                done = step == 9
                
                # Add transition
                agent.add_transition(state, action, reward, next_state, done)
                
                state = next_state
                
                if done:
                    break
            
            # Train world model periodically
            if episode % 2 == 0 and len(agent.transition_buffer) >= 32:
                metrics = agent.train_world_model(epochs=5, batch_size=32)
                assert isinstance(metrics, dict)
        
        # Final action selection should work
        final_state = np.random.rand(4)
        action = agent.select_action(final_state)
        assert 0 <= action < 12

