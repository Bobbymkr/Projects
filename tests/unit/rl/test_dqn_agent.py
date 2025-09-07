"""
Professional unit tests for DQN Agent implementation.

This test suite covers:
- Policy forward pass validation
- Action selection mechanisms  
- Training step mechanics
- Replay buffer functionality
- Network architecture validation
- Serialization and state management
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.rl.dqn_agent import DQNAgent, DQNConfig, QNet, ReplayBuffer, PrioritizedReplayBuffer
import tempfile
import os


class TestQNet:
    """Unit tests for the Q-Network architecture."""
    
    @pytest.fixture
    def qnet(self):
        return QNet(state_dim=4, action_dim=2, hidden=64, seed=42)
    
    def test_network_initialization(self, qnet):
        """Test network parameters are properly initialized."""
        assert qnet.W1.shape == (4, 64)
        assert qnet.b1.shape == (64,)
        assert qnet.W2.shape == (64, 64)
        assert qnet.b2.shape == (64,)
        assert qnet.W3.shape == (64, 2)
        assert qnet.b3.shape == (2,)
        
        # Check He initialization - weights should not be zeros
        assert not np.allclose(qnet.W1, 0)
        assert not np.allclose(qnet.W2, 0)
        assert not np.allclose(qnet.W3, 0)
        
        # Biases should be zero
        assert np.allclose(qnet.b1, 0)
        assert np.allclose(qnet.b2, 0)
        assert np.allclose(qnet.b3, 0)

    def test_forward_pass_shapes(self, qnet):
        """Test forward pass input/output shapes."""
        batch_size = 8
        state_dim = 4
        action_dim = 2
        
        states = np.random.randn(batch_size, state_dim)
        q_values, cache = qnet.forward(states)
        
        assert q_values.shape == (batch_size, action_dim)
        assert 'x' in cache
        assert 'z1' in cache
        assert 'a1' in cache
        assert 'z2' in cache
        assert 'a2' in cache
        
        # Check intermediate shapes
        assert cache['x'].shape == (batch_size, state_dim)
        assert cache['z1'].shape == (batch_size, 64)
        assert cache['a1'].shape == (batch_size, 64)

    def test_forward_pass_no_nans(self, qnet):
        """Test forward pass doesn't produce NaNs or Infs."""
        states = np.random.randn(5, 4)
        q_values, _ = qnet.forward(states)
        
        assert np.all(np.isfinite(q_values))
        assert not np.any(np.isnan(q_values))

    def test_backward_pass(self, qnet):
        """Test backward pass produces gradients of correct shapes."""
        states = np.random.randn(4, 4)
        q_values, cache = qnet.forward(states)
        
        # Simulate loss gradient
        dq = np.random.randn(*q_values.shape)
        grads = qnet.backward(cache, dq)
        
        # Check gradient shapes match parameter shapes
        assert len(grads) == len(qnet.params)
        for grad, param in zip(grads, qnet.params):
            assert grad.shape == param.shape
            
        # Gradients should not be all zeros (unless very specific case)
        assert not all(np.allclose(grad, 0) for grad in grads)

    def test_copy_from(self, qnet):
        """Test parameter copying between networks."""
        other_net = QNet(state_dim=4, action_dim=2, hidden=64, seed=123)
        
        # Verify networks are different initially
        assert not np.allclose(qnet.W1, other_net.W1)
        
        # Copy parameters
        qnet.copy_from(other_net)
        
        # Verify parameters are now identical
        for p1, p2 in zip(qnet.params, other_net.params):
            assert np.allclose(p1, p2)

    def test_save_load_consistency(self, qnet):
        """Test model saving and loading preserves parameters."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            try:
                # Save model
                qnet.save(tmp.name)
                
                # Create new network and load
                new_net = QNet(state_dim=4, action_dim=2, hidden=64)
                new_net.load(tmp.name)
                
                # Verify parameters match
                for p1, p2 in zip(qnet.params, new_net.params):
                    assert np.allclose(p1, p2)
            finally:
                os.unlink(tmp.name)


class TestReplayBuffer:
    """Unit tests for replay buffer functionality."""
    
    @pytest.fixture
    def buffer(self):
        return ReplayBuffer(capacity=1000, state_dim=4)
    
    def test_buffer_initialization(self, buffer):
        """Test buffer is properly initialized."""
        assert buffer.capacity == 1000
        assert buffer.states.shape == (1000, 4)
        assert buffer.actions.shape == (1000,)
        assert buffer.rewards.shape == (1000,)
        assert buffer.next_states.shape == (1000, 4)
        assert buffer.dones.shape == (1000,)
        assert buffer.ptr == 0
        assert buffer.size() == 0

    def test_add_experience(self, buffer):
        """Test adding experiences to buffer."""
        state = np.random.randn(4)
        action = 1
        reward = -0.5
        next_state = np.random.randn(4)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert buffer.size() == 1
        assert buffer.ptr == 1
        assert np.allclose(buffer.states[0], state)
        assert buffer.actions[0] == action
        assert buffer.rewards[0] == reward
        assert np.allclose(buffer.next_states[0], next_state)
        assert buffer.dones[0] == done

    def test_buffer_overflow(self, buffer):
        """Test buffer correctly handles overflow."""
        # Fill buffer beyond capacity
        for i in range(1500):  # More than capacity
            state = np.random.randn(4)
            buffer.add(state, i % 2, i, state, False)
        
        assert buffer.size() == buffer.capacity
        assert buffer.ptr == 500  # Should wrap around

    def test_sample_batch(self, buffer):
        """Test sampling batches from buffer."""
        # Add some experiences
        for i in range(100):
            state = np.random.randn(4)
            buffer.add(state, i % 2, i, state, i % 10 == 0)
        
        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 4)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 4)
        assert dones.shape == (batch_size,)


class TestDQNAgent:
    """Unit tests for DQN Agent."""
    
    @pytest.fixture
    def config(self):
        return DQNConfig(
            gamma=0.99,
            lr=1e-3,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=1000,
            batch_size=32,
            target_update=100,
            buffer_size=10000,
            warmup=100,
            seed=42
        )
    
    @pytest.fixture
    def agent(self, config):
        return DQNAgent(state_dim=4, action_dim=2, cfg=config)

    def test_agent_initialization(self, agent, config):
        """Test agent is properly initialized."""
        assert agent.cfg == config
        assert agent.action_dim == 2
        assert agent.steps == 0
        assert isinstance(agent.q, QNet)
        assert isinstance(agent.target, QNet)
        assert isinstance(agent.buffer, ReplayBuffer)
        
        # Target network should be copy of Q network initially
        for p1, p2 in zip(agent.q.params, agent.target.params):
            assert np.allclose(p1, p2)

    def test_action_selection_exploration(self, agent):
        """Test action selection during exploration."""
        state = np.random.randn(4)
        
        # At start, epsilon should be high, so mostly random actions
        actions = []
        for _ in range(100):
            action = agent.select_action(state, evaluate=False)
            actions.append(action)
            assert 0 <= action < agent.action_dim
        
        # Should have some variety in actions due to exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_action_selection_evaluation(self, agent):
        """Test action selection during evaluation (no exploration)."""
        state = np.random.randn(4)
        
        # In evaluation mode, should be deterministic
        actions = []
        for _ in range(10):
            action = agent.select_action(state, evaluate=True)
            actions.append(action)
        
        # All actions should be the same (deterministic)
        assert len(set(actions)) == 1

    def test_epsilon_decay(self, agent):
        """Test epsilon decay schedule."""
        initial_eps = agent._epsilon()
        
        # Advance steps
        agent.steps = 500
        mid_eps = agent._epsilon()
        
        agent.steps = 2000
        final_eps = agent._epsilon()
        
        # Epsilon should decrease over time
        assert initial_eps > mid_eps > final_eps
        assert final_eps >= agent.cfg.eps_end

    def test_experience_storage(self, agent):
        """Test storing experiences in replay buffer."""
        state = np.random.randn(4)
        action = 1
        reward = -0.1
        next_state = np.random.randn(4)
        done = False
        
        initial_size = agent.buffer.size()
        agent.push(state, action, reward, next_state, done)
        
        assert agent.buffer.size() == initial_size + 1

    def test_training_step_warmup(self, agent):
        """Test training step during warmup period."""
        # During warmup, should not train
        loss = agent.train_step()
        assert loss is None
        assert agent.steps == 1

    def test_training_step_with_data(self, agent):
        """Test training step with sufficient data."""
        # Fill buffer with experiences
        for i in range(agent.cfg.warmup + agent.cfg.batch_size):
            state = np.random.randn(4)
            action = i % agent.action_dim
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 20 == 0
            agent.push(state, action, reward, next_state, done)
        
        # Now training step should work
        initial_params = [p.copy() for p in agent.q.params]
        loss = agent.train_step()
        
        assert loss is not None
        assert isinstance(loss, float)
        assert np.isfinite(loss)
        
        # Parameters should have changed
        params_changed = any(
            not np.allclose(p1, p2) 
            for p1, p2 in zip(initial_params, agent.q.params)
        )
        assert params_changed

    def test_target_network_update(self, agent):
        """Test target network periodic update."""
        # Fill buffer and advance to target update step
        for i in range(agent.cfg.warmup + agent.cfg.batch_size):
            state = np.random.randn(4)
            agent.push(state, i % 2, np.random.randn(), state, False)
        
        # Modify Q network parameters slightly
        agent.q.W1 += 0.01
        initial_target_W1 = agent.target.W1.copy()
        
        # Set steps to trigger target update
        agent.steps = agent.cfg.target_update - 1
        
        # Train step should update target network
        agent.train_step()
        
        # Target network should now match Q network
        assert not np.allclose(agent.target.W1, initial_target_W1)
        assert np.allclose(agent.target.W1, agent.q.W1)

    def test_save_load_agent(self, agent):
        """Test agent model saving and loading."""
        # Train agent a bit to change parameters
        for i in range(150):  # Past warmup
            state = np.random.randn(4)
            agent.push(state, i % 2, np.random.randn(), state, False)
            if i >= agent.cfg.warmup:
                agent.train_step()
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            try:
                # Save agent
                original_params = [p.copy() for p in agent.q.params]
                agent.save(tmp.name)
                
                # Create new agent and load
                new_agent = DQNAgent(state_dim=4, action_dim=2, cfg=agent.cfg)
                new_agent.load(tmp.name)
                
                # Parameters should match
                for p1, p2 in zip(original_params, new_agent.q.params):
                    assert np.allclose(p1, p2)
                
                # Target network should match too
                for p1, p2 in zip(new_agent.q.params, new_agent.target.params):
                    assert np.allclose(p1, p2)
                
            finally:
                os.unlink(tmp.name)

    def test_deterministic_seeding(self):
        """Test that seeding produces deterministic behavior."""
        config1 = DQNConfig(seed=42)
        config2 = DQNConfig(seed=42)
        
        agent1 = DQNAgent(state_dim=4, action_dim=2, cfg=config1)
        agent2 = DQNAgent(state_dim=4, action_dim=2, cfg=config2)
        
        # Same seed should produce same initial parameters
        for p1, p2 in zip(agent1.q.params, agent2.q.params):
            assert np.allclose(p1, p2)
        
        # Same actions on same state
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action1 = agent1.select_action(state, evaluate=True)
        action2 = agent2.select_action(state, evaluate=True)
        assert action1 == action2


class TestPrioritizedReplayBuffer:
    """Unit tests for prioritized replay buffer."""
    
    @pytest.fixture
    def priority_buffer(self):
        return PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    
    def test_priority_buffer_initialization(self, priority_buffer):
        """Test prioritized buffer initialization."""
        assert priority_buffer.capacity == 1000
        assert priority_buffer.alpha == 0.6
        assert priority_buffer.beta == 0.4
        assert len(priority_buffer) == 0

    def test_add_experience_with_priority(self, priority_buffer):
        """Test adding experiences with priority calculation."""
        experience = (np.random.randn(4), 1, -0.5, np.random.randn(4), False)
        priority_buffer.add(experience)
        
        assert len(priority_buffer) == 1

    def test_sample_with_importance_weights(self, priority_buffer):
        """Test sampling with importance sampling weights."""
        # Add experiences
        for i in range(100):
            exp = (np.random.randn(4), i % 2, i, np.random.randn(4), i % 10 == 0)
            priority_buffer.add(exp)
        
        # Sample batch
        batch_size = 32
        result = priority_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, idxs, weights = result
        
        assert len(states) == batch_size
        assert len(weights) == batch_size
        assert all(w > 0 for w in weights)  # Importance weights should be positive

    def test_priority_update(self, priority_buffer):
        """Test updating priorities based on TD errors."""
        # Add some experiences
        for i in range(50):
            exp = (np.random.randn(4), i % 2, i, np.random.randn(4), False)
            priority_buffer.add(exp)
        
        # Sample and update priorities
        states, actions, rewards, next_states, dones, idxs, weights = priority_buffer.sample(10)
        
        # Update with mock TD errors
        for idx in idxs:
            td_error = np.random.random()
            priority_buffer.update(idx, td_error)
        
        # Buffer should handle updates without errors
        assert len(priority_buffer) == 50


@pytest.mark.perf
class TestDQNPerformance:
    """Performance tests for DQN components."""
    
    def test_forward_pass_performance(self):
        """Test forward pass performance."""
        import time
        
        qnet = QNet(state_dim=84, action_dim=4, hidden=512)  # Larger network
        states = np.random.randn(1000, 84)  # Large batch
        
        start_time = time.time()
        q_values, _ = qnet.forward(states)
        end_time = time.time()
        
        inference_time = end_time - start_time
        throughput = len(states) / inference_time
        
        # Should process at least 1000 samples per second
        assert throughput > 1000, f"Throughput too low: {throughput:.1f} samples/sec"
        
    def test_training_step_performance(self):
        """Test training step performance."""
        import time
        
        config = DQNConfig(batch_size=64, buffer_size=50000)
        agent = DQNAgent(state_dim=84, action_dim=4, cfg=config)
        
        # Fill buffer
        for i in range(config.warmup + 100):
            state = np.random.randn(84)
            agent.push(state, i % 4, np.random.randn(), state, i % 100 == 0)
        
        # Time training steps
        start_time = time.time()
        for _ in range(100):
            agent.train_step()
        end_time = time.time()
        
        avg_step_time = (end_time - start_time) / 100
        
        # Training step should be reasonably fast
        assert avg_step_time < 0.1, f"Training step too slow: {avg_step_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
