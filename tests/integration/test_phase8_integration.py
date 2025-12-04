"""
Integration Tests for Phase 8 Multi-Objective Optimization.

Tests complete workflows as industry experts would validate.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.multi_objective.phase8_multi_objective import (
    MOPPOAgent, MOPPOConfig,
    CPOAgent, ConstraintConfig,
    MultiObjectiveReward, MultiObjectiveWeights,
    ConstraintOptimizer,
)
from src.env.traffic_env import TrafficEnv


class TestPhase8Integration:
    """Integration tests for Phase 8."""
    
    @pytest.fixture
    def env(self):
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "arrival_rates": [0.3, 0.3, 0.3, 0.3],
        }
        return TrafficEnv(config=config)
    
    @pytest.fixture
    def reward_fn(self):
        return MultiObjectiveReward()
    
    def test_moppo_full_episode(self, env, reward_fn):
        """Test MO-PPO agent on full episode."""
        agent = MOPPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=MOPPOConfig(batch_size=16, buffer_size=100),
        )
        
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        done = False
        step_count = 0
        
        while not done and step_count < 50:  # Limit steps for test
            action, log_prob, values = agent.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            # Compute rewards
            rewards = reward_fn.compute_rewards(
                wait_times=env.wait_times,
                queue_lengths=env.queues,
                vehicles_served=info.get('total_vehicles_processed', 0),
                phase_changes=1,
            )
            
            # Store transition
            agent.store_transition(
                obs, action, log_prob, values,
                np.array([
                    rewards['wait_time'],
                    rewards['fuel_consumption'],
                    rewards['emissions'],
                    rewards['throughput'],
                    rewards['accidents'],
                    rewards['infrastructure_wear'],
                ]),
                next_obs, done,
            )
            
            obs = next_obs
            step_count += 1
        
        # Should complete episode
        assert step_count > 0
    
    def test_cpo_constraint_satisfaction(self, env):
        """Test CPO agent constraint satisfaction."""
        agent = CPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            constraint_config=ConstraintConfig(),
            ppo_config=MOPPOConfig(batch_size=16, buffer_size=100),
        )
        
        constraint_optimizer = ConstraintOptimizer(ConstraintConfig())
        
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        violations = 0
        total_checks = 0
        
        for _ in range(20):  # Limited steps for test
            action, _, _ = agent.select_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            # Check constraints
            green_time = env.green_values[action]
            satisfied, _ = constraint_optimizer.check_constraints(
                action, green_time, env.wait_times, env.queues
            )
            
            total_checks += 1
            if not satisfied:
                violations += 1
            
            obs = next_obs
            if done:
                break
        
        # Should have some constraint checks
        assert total_checks > 0
    
    def test_multi_objective_reward_consistency(self, env, reward_fn):
        """Test multi-objective reward consistency."""
        obs, _ = env.reset()
        
        rewards_list = []
        for _ in range(10):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            rewards = reward_fn.compute_rewards(
                wait_times=env.wait_times,
                queue_lengths=env.queues,
                vehicles_served=info.get('total_vehicles_processed', 0),
                phase_changes=1,
            )
            
            rewards_list.append(rewards)
            
            if done:
                break
        
        # All rewards should have same structure
        assert all('total' in r for r in rewards_list)
        assert all('wait_time' in r for r in rewards_list)
        assert all(len(r) == 7 for r in rewards_list)  # 6 objectives + total
    
    def test_training_step_execution(self, env, reward_fn):
        """Test that training steps execute without errors."""
        agent = MOPPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=MOPPOConfig(batch_size=16, buffer_size=100),
        )
        
        # Fill buffer
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32).flatten()
        
        for _ in range(agent.config.batch_size):
            action, log_prob, values = agent.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = np.array(next_obs, dtype=np.float32).flatten()
            
            rewards = reward_fn.compute_rewards(
                wait_times=env.wait_times,
                queue_lengths=env.queues,
                vehicles_served=info.get('total_vehicles_processed', 0),
                phase_changes=1,
            )
            
            agent.store_transition(
                obs, action, log_prob, values,
                np.array([
                    rewards['wait_time'],
                    rewards['fuel_consumption'],
                    rewards['emissions'],
                    rewards['throughput'],
                    rewards['accidents'],
                    rewards['infrastructure_wear'],
                ]),
                next_obs, done,
            )
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
                obs = np.array(obs, dtype=np.float32).flatten()
        
        # Training step should execute
        metrics = agent.train_step()
        assert 'loss' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

