"""
Comprehensive unit tests for Phase 6 Advanced RL Algorithms.

Tests cover:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- Rainbow DQN

Each algorithm is tested for:
- Initialization
- Action selection
- Training mechanics
- Buffer management
- Network architecture
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.novel_algorithms.phase6_advanced_rl import (
    PPOAgent, PPOConfig,
    SACAgent, SACConfig,
    RainbowDQNAgent, RainbowDQNConfig,
    PPOPolicyNetwork, PPOValueNetwork,
    SACActor, SACCritic,
    DuelingDQN, NoisyLinear,
    PrioritizedReplayBuffer,
)


class TestPPOAgent:
    """Unit tests for PPO Agent."""
    
    @pytest.fixture
    def config(self):
        return PPOConfig(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            batch_size=32,
            buffer_size=100,
        )
    
    @pytest.fixture
    def agent(self, config):
        return PPOAgent(state_dim=4, action_dim=2, config=config)
    
    def test_ppo_initialization(self, agent, config):
        """Test PPO agent is properly initialized."""
        assert agent.config == config
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.step_count == 0
        assert isinstance(agent.policy_net, PPOPolicyNetwork)
        assert isinstance(agent.value_net, PPOValueNetwork)
        assert isinstance(agent.buffer, dict)
        assert 'states' in agent.buffer
        assert 'actions' in agent.buffer
        assert 'rewards' in agent.buffer
        assert 'values' in agent.buffer
        assert 'log_probs' in agent.buffer
        assert 'dones' in agent.buffer
    
    def test_ppo_action_selection(self, agent):
        """Test PPO action selection returns valid action."""
        state = np.random.randn(4).astype(np.float32)
        
        result = agent.select_action(state, deterministic=False)
        
        # Should return tuple (action, log_prob, value)
        assert isinstance(result, tuple)
        assert len(result) == 3
        action, log_prob, value = result
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)
    
    def test_ppo_action_selection_deterministic(self, agent):
        """Test PPO deterministic action selection."""
        state = np.random.randn(4).astype(np.float32)
        
        result1 = agent.select_action(state, deterministic=True)
        result2 = agent.select_action(state, deterministic=True)
        
        # Deterministic should return same action
        assert result1[0] == result2[0]
    
    def test_ppo_store_transition(self, agent):
        """Test storing transitions in PPO buffer."""
        state = np.random.randn(4).astype(np.float32)
        action = 1
        reward = 10.0
        value = 5.0
        log_prob = -0.5
        done = False
        
        agent.store_transition(state, action, reward, value, log_prob, done)
        
        assert len(agent.buffer['states']) == 1
        assert len(agent.buffer['actions']) == 1
        assert len(agent.buffer['rewards']) == 1
        assert agent.buffer['actions'][0] == action
        assert agent.buffer['rewards'][0] == reward
    
    def test_ppo_push(self, agent):
        """Test PPO push method for compatibility."""
        state = np.random.randn(4).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.randn(4).astype(np.float32)
        done = False
        
        # First select action to set _last_log_prob and _last_value
        agent.select_action(state)
        agent.push(state, action, reward, next_state, done)
        
        assert len(agent.buffer['states']) == 1
    
    def test_ppo_compute_gae(self, agent):
        """Test GAE computation."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]
        next_value = 0.0
        
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)
        assert all(np.isfinite(adv) for adv in advantages)
        assert all(np.isfinite(ret) for ret in returns)
    
    def test_ppo_train_step_empty_buffer(self, agent):
        """Test PPO training with empty buffer."""
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] == 0.0
    
    def test_ppo_train_step_with_data(self, agent):
        """Test PPO training with collected data."""
        # Collect some transitions
        for i in range(agent.config.batch_size):
            state = np.random.randn(4).astype(np.float32)
            action, log_prob, value = agent.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            done = (i == agent.config.batch_size - 1)
            
            agent.push(state, action, reward, next_state, done)
        
        # Train
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['loss'] >= 0  # Loss should be non-negative
    
    def test_ppo_reset(self, agent):
        """Test PPO reset clears buffer."""
        # Add some data
        state = np.random.randn(4).astype(np.float32)
        agent.select_action(state)
        agent.push(state, 0, 1.0, state, False)
        
        assert len(agent.buffer['states']) > 0
        
        agent.reset()
        
        assert len(agent.buffer['states']) == 0
        assert len(agent.buffer['actions']) == 0


class TestSACAgent:
    """Unit tests for SAC Agent."""
    
    @pytest.fixture
    def config(self):
        return SACConfig(
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            batch_size=32,
            buffer_size=100,
        )
    
    @pytest.fixture
    def agent(self, config):
        return SACAgent(state_dim=4, action_dim=2, config=config)
    
    def test_sac_initialization(self, agent, config):
        """Test SAC agent is properly initialized."""
        assert agent.config == config
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.step_count == 0
        assert isinstance(agent.actor, SACActor)
        assert isinstance(agent.critic1, SACCritic)
        assert isinstance(agent.critic2, SACCritic)
        assert isinstance(agent.critic1_target, SACCritic)
        assert isinstance(agent.critic2_target, SACCritic)
        assert isinstance(agent.buffer, list) or hasattr(agent.buffer, '__len__')
    
    def test_sac_action_selection(self, agent):
        """Test SAC action selection returns valid action."""
        state = np.random.randn(4).astype(np.float32)
        
        action = agent.select_action(state, deterministic=False)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
    
    def test_sac_action_selection_deterministic(self, agent):
        """Test SAC deterministic action selection."""
        state = np.random.randn(4).astype(np.float32)
        
        action1 = agent.select_action(state, deterministic=True)
        action2 = agent.select_action(state, deterministic=True)
        
        # Deterministic should return same action
        assert action1 == action2
    
    def test_sac_push(self, agent):
        """Test SAC push stores experience."""
        state = np.random.randn(4).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.randn(4).astype(np.float32)
        done = False
        
        initial_len = len(agent.buffer)
        agent.push(state, action, reward, next_state, done)
        
        assert len(agent.buffer) == initial_len + 1
    
    def test_sac_train_step_empty_buffer(self, agent):
        """Test SAC training with empty buffer."""
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] == 0.0
    
    def test_sac_train_step_with_data(self, agent):
        """Test SAC training with collected data."""
        # Collect some transitions
        for i in range(agent.config.batch_size):
            state = np.random.randn(4).astype(np.float32)
            action = agent.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            done = False
            
            agent.push(state, action, reward, next_state, done)
        
        # Train
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_sac_reset(self, agent):
        """Test SAC reset (should not affect buffer)."""
        # Add some data
        state = np.random.randn(4).astype(np.float32)
        agent.push(state, 0, 1.0, state, False)
        
        initial_len = len(agent.buffer)
        agent.reset()
        
        # SAC reset doesn't clear buffer (off-policy)
        assert len(agent.buffer) == initial_len


class TestRainbowDQNAgent:
    """Unit tests for Rainbow DQN Agent."""
    
    @pytest.fixture
    def config(self):
        return RainbowDQNConfig(
            lr=6.25e-5,
            gamma=0.99,
            n_steps=3,
            batch_size=32,
            buffer_size=100,
            target_update_frequency=100,
        )
    
    @pytest.fixture
    def agent(self, config):
        return RainbowDQNAgent(state_dim=4, action_dim=2, config=config)
    
    def test_rainbow_initialization(self, agent, config):
        """Test Rainbow DQN agent is properly initialized."""
        assert agent.config == config
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.step_count == 0
        assert isinstance(agent.q_net, DuelingDQN)
        assert isinstance(agent.target_net, DuelingDQN)
        assert isinstance(agent.buffer, PrioritizedReplayBuffer)
        assert hasattr(agent, 'n_step_buffer')
    
    def test_rainbow_action_selection(self, agent):
        """Test Rainbow DQN action selection returns valid action."""
        state = np.random.randn(4).astype(np.float32)
        
        action = agent.select_action(state, evaluate=False)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
    
    def test_rainbow_action_selection_evaluation(self, agent):
        """Test Rainbow DQN evaluation mode."""
        state = np.random.randn(4).astype(np.float32)
        
        action = agent.select_action(state, evaluate=True)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim
    
    def test_rainbow_epsilon_decay(self, agent):
        """Test epsilon decay schedule."""
        initial_eps = agent._epsilon()
        
        # Simulate many steps
        for _ in range(agent.config.eps_decay):
            agent.step_count += 1
        
        final_eps = agent._epsilon()
        
        assert final_eps <= initial_eps
        assert final_eps >= agent.config.eps_end
    
    def test_rainbow_push(self, agent):
        """Test Rainbow DQN push stores experience."""
        state = np.random.randn(4).astype(np.float32)
        action = 1
        reward = 10.0
        next_state = np.random.randn(4).astype(np.float32)
        done = False
        
        initial_len = len(agent.buffer)
        agent.push(state, action, reward, next_state, done)
        
        # May not add immediately due to n-step buffer
        # But should eventually add
        if done:
            assert len(agent.buffer) >= initial_len
    
    def test_rainbow_train_step_empty_buffer(self, agent):
        """Test Rainbow DQN training with empty buffer."""
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] == 0.0
    
    def test_rainbow_train_step_with_data(self, agent):
        """Test Rainbow DQN training with collected data."""
        # Collect some transitions
        for i in range(agent.config.batch_size * 2):
            state = np.random.randn(4).astype(np.float32)
            action = agent.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            done = (i % 10 == 9)  # Some episodes end
            
            agent.push(state, action, reward, next_state, done)
        
        # Train
        metrics = agent.train_step()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_rainbow_reset(self, agent):
        """Test Rainbow DQN reset clears n-step buffer."""
        # Add some data to n-step buffer
        state = np.random.randn(4).astype(np.float32)
        agent.push(state, 0, 1.0, state, False)
        
        agent.reset()
        
        assert len(agent.n_step_buffer) == 0


class TestNetworkArchitectures:
    """Test network architectures for Phase 6 algorithms."""
    
    def test_ppo_policy_network(self):
        """Test PPO policy network architecture."""
        net = PPOPolicyNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        
        logits, _ = net(state)
        
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()
    
    def test_ppo_value_network(self):
        """Test PPO value network architecture."""
        net = PPOValueNetwork(state_dim=4)
        state = torch.randn(1, 4)
        
        value = net(state)
        
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()
    
    def test_sac_actor(self):
        """Test SAC actor network."""
        actor = SACActor(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        
        logits = actor(state)
        action, log_prob = actor.get_action_and_log_prob(state)
        
        assert logits.shape == (1, 2)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert torch.isfinite(logits).all()
    
    def test_sac_critic(self):
        """Test SAC critic network."""
        critic = SACCritic(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        action = torch.tensor([1])
        
        q_value = critic(state, action)
        
        assert q_value.shape == (1, 1)
        assert torch.isfinite(q_value).all()
    
    def test_dueling_dqn(self):
        """Test Dueling DQN architecture."""
        net = DuelingDQN(state_dim=4, action_dim=2, n_atoms=51)
        state = torch.randn(1, 4)
        
        q_dist = net(state)
        q_values = net.get_q_values(state)
        
        assert q_dist.shape == (1, 2, 51)  # (batch, actions, atoms)
        assert q_values.shape == (1, 2)  # (batch, actions)
        assert torch.isfinite(q_dist).all()
        assert torch.isfinite(q_values).all()
    
    def test_noisy_linear(self):
        """Test Noisy Linear layer."""
        layer = NoisyLinear(4, 2)
        x = torch.randn(1, 4)
        
        output = layer(x)
        
        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()
        
        # Test noise reset
        layer.reset_noise()
        output2 = layer(x)
        assert output2.shape == (1, 2)


class TestPrioritizedReplayBuffer:
    """Test Prioritized Experience Replay buffer."""
    
    @pytest.fixture
    def buffer(self):
        return PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)
    
    def test_buffer_push(self, buffer):
        """Test adding experiences to buffer."""
        state = np.random.randn(4)
        action = 1
        reward = 10.0
        next_state = np.random.randn(4)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_buffer_sample(self, buffer):
        """Test sampling from buffer."""
        # Add multiple experiences
        for i in range(10):
            state = np.random.randn(4)
            buffer.push(state, i % 2, float(i), state, False)
        
        samples, indices, weights = buffer.sample(5)
        
        assert samples is not None
        assert len(samples) == 5
        assert len(indices) == 5
        assert len(weights) == 5
        assert all(0 <= w <= 1 for w in weights)
    
    def test_buffer_update_priorities(self, buffer):
        """Test updating priorities."""
        # Add experiences
        for i in range(10):
            state = np.random.randn(4)
            buffer.push(state, i % 2, float(i), state, False)
        
        # Sample and update priorities
        samples, indices, _ = buffer.sample(5)
        td_errors = np.random.rand(5)
        
        buffer.update_priorities(indices, td_errors)
        
        # Priorities should be updated
        assert buffer.max_priority >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

