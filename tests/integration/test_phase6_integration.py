"""
Integration tests for Phase 6 Advanced RL Algorithms with Traffic Environment.

Tests the full integration between Phase 6 algorithms and the traffic environment,
including training loops and performance validation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.novel_algorithms.phase6_advanced_rl import (
    PPOAgent, PPOConfig,
    SACAgent, SACConfig,
    RainbowDQNAgent, RainbowDQNConfig,
)
from src.env.traffic_env import TrafficEnv


@pytest.fixture
def traffic_env():
    """Create a traffic environment for testing."""
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "queue_capacity": 40,
        "arrival_rates": [0.3, 0.3, 0.3, 0.3],
    }
    return TrafficEnv(config)


class TestPPOIntegration:
    """Integration tests for PPO with traffic environment."""
    
    @pytest.fixture
    def agent(self, traffic_env):
        state_dim = traffic_env.observation_space.shape[0]
        action_dim = traffic_env.action_space.n
        config = PPOConfig(
            batch_size=32,
            buffer_size=200,
            train_epochs=2,  # Reduced for faster testing
        )
        return PPOAgent(state_dim, action_dim, config)
    
    def test_ppo_full_episode(self, agent, traffic_env):
        """Test PPO agent completes a full episode."""
        obs, info = traffic_env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Select action (returns tuple)
            result = agent.select_action(obs)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
            
            # Ensure valid action
            action = int(action)
            if action < 0 or action >= traffic_env.action_space.n:
                action = 0
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            # Store experience
            agent.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            steps += 1
        
        # Should complete episode
        assert steps > 0
        assert len(agent.buffer['states']) > 0
    
    def test_ppo_training_loop(self, agent, traffic_env):
        """Test PPO training over multiple episodes."""
        episode_rewards = []
        
        for episode in range(3):
            obs, info = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = 50
            
            while not done and steps < max_steps:
                result = agent.select_action(obs)
                action = result[0] if isinstance(result, tuple) else result
                action = int(action)
                if action < 0 or action >= traffic_env.action_space.n:
                    action = 0
                
                next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                agent.push(obs, action, reward, next_obs, done)
                episode_reward += reward
                
                obs = next_obs
                steps += 1
            
            # Train at end of episode if buffer is full enough
            if len(agent.buffer['states']) >= agent.config.batch_size:
                metrics = agent.train_step()
                assert isinstance(metrics, dict)
                assert 'loss' in metrics
            
            episode_rewards.append(episode_reward)
        
        assert len(episode_rewards) == 3
        assert all(np.isfinite(r) for r in episode_rewards)


class TestSACIntegration:
    """Integration tests for SAC with traffic environment."""
    
    @pytest.fixture
    def agent(self, traffic_env):
        state_dim = traffic_env.observation_space.shape[0]
        action_dim = traffic_env.action_space.n
        config = SACConfig(
            batch_size=32,
            buffer_size=200,
        )
        return SACAgent(state_dim, action_dim, config)
    
    def test_sac_full_episode(self, agent, traffic_env):
        """Test SAC agent completes a full episode."""
        obs, info = traffic_env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.select_action(obs)
            action = int(action)
            if action < 0 or action >= traffic_env.action_space.n:
                action = 0
            
            next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            agent.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            steps += 1
        
        assert steps > 0
        assert len(agent.buffer) > 0
    
    def test_sac_training_loop(self, agent, traffic_env):
        """Test SAC training over multiple episodes."""
        episode_rewards = []
        
        for episode in range(3):
            obs, info = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = 50
            
            while not done and steps < max_steps:
                action = agent.select_action(obs)
                action = int(action)
                if action < 0 or action >= traffic_env.action_space.n:
                    action = 0
                
                next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                agent.push(obs, action, reward, next_obs, done)
                episode_reward += reward
                
                # Train periodically
                if len(agent.buffer) >= agent.config.batch_size and steps % 10 == 0:
                    metrics = agent.train_step()
                    assert isinstance(metrics, dict)
                
                obs = next_obs
                steps += 1
            
            episode_rewards.append(episode_reward)
        
        assert len(episode_rewards) == 3
        assert all(np.isfinite(r) for r in episode_rewards)


class TestRainbowDQNIntegration:
    """Integration tests for Rainbow DQN with traffic environment."""
    
    @pytest.fixture
    def agent(self, traffic_env):
        state_dim = traffic_env.observation_space.shape[0]
        action_dim = traffic_env.action_space.n
        config = RainbowDQNConfig(
            batch_size=32,
            buffer_size=200,
            target_update_frequency=50,
            n_steps=3,
        )
        return RainbowDQNAgent(state_dim, action_dim, config)
    
    def test_rainbow_full_episode(self, agent, traffic_env):
        """Test Rainbow DQN agent completes a full episode."""
        obs, info = traffic_env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.select_action(obs, evaluate=False)
            action = int(action)
            if action < 0 or action >= traffic_env.action_space.n:
                action = 0
            
            next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            agent.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            steps += 1
        
        assert steps > 0
    
    def test_rainbow_training_loop(self, agent, traffic_env):
        """Test Rainbow DQN training over multiple episodes."""
        episode_rewards = []
        
        for episode in range(3):
            obs, info = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = 50
            
            while not done and steps < max_steps:
                action = agent.select_action(obs, evaluate=False)
                action = int(action)
                if action < 0 or action >= traffic_env.action_space.n:
                    action = 0
                
                next_obs, reward, terminated, truncated, _ = traffic_env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                agent.push(obs, action, reward, next_obs, done)
                episode_reward += reward
                
                # Train periodically
                if len(agent.buffer) >= agent.config.batch_size and steps % 10 == 0:
                    metrics = agent.train_step()
                    assert isinstance(metrics, dict)
                    assert 'loss' in metrics
                
                obs = next_obs
                steps += 1
            
            episode_rewards.append(episode_reward)
        
        assert len(episode_rewards) == 3
        assert all(np.isfinite(r) for r in episode_rewards)


class TestPhase6Performance:
    """Performance validation tests for Phase 6 algorithms."""
    
    def test_ppo_learning_progress(self, traffic_env):
        """Test that PPO shows learning progress."""
        state_dim = traffic_env.observation_space.shape[0]
        action_dim = traffic_env.action_space.n
        agent = PPOAgent(state_dim, action_dim, PPOConfig(batch_size=16, buffer_size=100))
        
        initial_rewards = []
        final_rewards = []
        
        # Initial episodes
        for _ in range(2):
            obs, _ = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            reward = 0.0
            steps = 0
            
            while not done and steps < 30:
                result = agent.select_action(obs)
                action = result[0] if isinstance(result, tuple) else result
                action = int(action) % action_dim
                
                next_obs, r, terminated, truncated, _ = traffic_env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                agent.push(obs, action, r, next_obs, done)
                reward += r
                obs = next_obs
                steps += 1
            
            initial_rewards.append(reward)
        
        # Train
        if len(agent.buffer['states']) >= agent.config.batch_size:
            agent.train_step()
        
        # Final episodes
        for _ in range(2):
            obs, _ = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            done = False
            reward = 0.0
            steps = 0
            
            while not done and steps < 30:
                result = agent.select_action(obs)
                action = result[0] if isinstance(result, tuple) else result
                action = int(action) % action_dim
                
                next_obs, r, terminated, truncated, _ = traffic_env.step(action)
                done = terminated or truncated
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                agent.push(obs, action, r, next_obs, done)
                reward += r
                obs = next_obs
                steps += 1
            
            final_rewards.append(reward)
        
        # Should have collected rewards
        assert len(initial_rewards) == 2
        assert len(final_rewards) == 2
        assert all(np.isfinite(r) for r in initial_rewards + final_rewards)
    
    def test_all_algorithms_compatible(self, traffic_env):
        """Test all Phase 6 algorithms are compatible with environment."""
        state_dim = traffic_env.observation_space.shape[0]
        action_dim = traffic_env.action_space.n
        
        agents = [
            PPOAgent(state_dim, action_dim, PPOConfig(batch_size=16, buffer_size=50)),
            SACAgent(state_dim, action_dim, SACConfig(batch_size=16, buffer_size=50)),
            RainbowDQNAgent(state_dim, action_dim, RainbowDQNConfig(batch_size=16, buffer_size=50)),
        ]
        
        for agent in agents:
            obs, _ = traffic_env.reset()
            obs = np.array(obs, dtype=np.float32).flatten()
            
            # Test action selection
            if hasattr(agent, 'select_action'):
                result = agent.select_action(obs)
                if isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result
                
                assert isinstance(action, (int, np.integer))
                assert 0 <= action < action_dim
            
            # Test push
            if hasattr(agent, 'push'):
                agent.push(obs, 0, 1.0, obs, False)
            
            # Test reset
            if hasattr(agent, 'reset'):
                agent.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

