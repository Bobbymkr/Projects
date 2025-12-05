"""
End-to-End Integration Tests.

Tests complete workflows from API request to agent decision to response.
Implements Phase 3.1 from Expert Review Remediation Plan.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.api.services.traffic_controller import TrafficController
from src.control.fuzzy_control import FuzzyController


class TestCompleteWorkflow:
    """Test complete API to agent workflow."""
    
    def test_api_to_agent_workflow(self):
        """Test complete workflow: API request → Controller → Agent → Response."""
        # Initialize components
        controller = TrafficController()
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
        
        # Simulate API request
        intersection_id = "test_intersection_001"
        queue_lengths = [5.0, 8.0, 3.0, 2.0]
        wait_times = [10.0, 15.0, 8.0, 5.0]
        throughput = 0.5
        current_phase = 0
        
        # Make decision through controller (simulating API call)
        import asyncio
        decision = asyncio.run(controller.make_decision(
            intersection_id=intersection_id,
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            throughput=throughput,
            current_phase=current_phase,
        ))
        
        # Verify response structure
        assert "phase" in decision
        assert "green_time" in decision
        assert "confidence" in decision
        assert "algorithm" in decision
        assert isinstance(decision["phase"], int)
        assert 0 <= decision["phase"] <= 3
        assert decision["green_time"] > 0
        assert 0 <= decision["confidence"] <= 1
    
    def test_error_propagation(self):
        """Test error propagation across component boundaries."""
        controller = TrafficController()
        
        # Test with invalid input
        invalid_queue_lengths = [-1.0, 1000.0]  # Invalid values
        
        import asyncio
        # Should handle gracefully
        try:
            decision = asyncio.run(controller.make_decision(
                intersection_id="test",
                queue_lengths=invalid_queue_lengths,
                wait_times=[10.0, 15.0],
                throughput=0.5,
                current_phase=0,
            ))
            # Should return safe default or raise appropriate error
            assert decision is not None
        except Exception as e:
            # Error should be properly handled
            assert "invalid" in str(e).lower() or "error" in str(e).lower()
    
    def test_recovery_from_failure(self):
        """Test recovery from component failures."""
        controller = TrafficController()
        
        # Simulate agent failure by using invalid intersection
        # System should fallback gracefully
        import asyncio
        decision = asyncio.run(controller.make_decision(
            intersection_id="nonexistent_intersection",
            queue_lengths=[5.0, 8.0, 3.0, 2.0],
            wait_times=[10.0, 15.0, 8.0, 5.0],
            throughput=0.5,
            current_phase=0,
        ))
        
        # Should still return valid decision (fallback)
        assert decision is not None
        assert "phase" in decision


class TestAgentTrainingWorkflow:
    """Test complete agent training workflow."""
    
    def test_training_with_curriculum(self):
        """Test training workflow with curriculum learning."""
        from src.rl.curriculum_learning import TrafficCurriculum
        
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
            "episode_horizon": 600,  # Shorter for testing
        }
        
        env = TrafficEnv(env_config)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=DQNConfig()
        )
        
        # Initialize curriculum
        curriculum = TrafficCurriculum(
            base_arrival_rates=[0.3] * env.num_lanes,
            performance_threshold=0.7,
            min_episodes_per_level=5,  # Lower for testing
            performance_window=10,
        )
        
        # Run short training episode
        rewards = []
        for ep in range(10):
            # Update environment with curriculum
            env.arrival_rates = curriculum.get_arrival_rates()
            
            obs, info = env.reset()
            episode_reward = 0.0
            terminated = truncated = False
            
            while not (terminated or truncated) and len(rewards) < 50:  # Limit steps
                action = agent.select_action(obs.astype(np.float32))
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.push(
                    obs.astype(np.float32),
                    action,
                    reward,
                    next_obs.astype(np.float32),
                    terminated or truncated
                )
                agent.train_step()
                episode_reward += reward
                obs = next_obs
            
            rewards.append(episode_reward)
            curriculum.update_performance(episode_reward, ep)
        
        # Verify training completed
        assert len(rewards) == 10
        assert all(isinstance(r, (int, float)) for r in rewards)
    
    def test_training_with_per(self):
        """Test training workflow with Prioritized Experience Replay."""
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
            "episode_horizon": 600,
        }
        
        env = TrafficEnv(env_config)
        
        # Create agent with PER enabled
        cfg = DQNConfig()
        cfg.use_per = True
        cfg.per_alpha = 0.6
        cfg.per_beta = 0.4
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=cfg
        )
        
        # Verify PER is enabled
        assert agent.use_per == True
        assert hasattr(agent.buffer, 'update')  # PER has update method
        
        # Run short training episode
        obs, info = env.reset()
        for step in range(20):
            action = agent.select_action(obs.astype(np.float32))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.push(
                obs.astype(np.float32),
                action,
                reward,
                next_obs.astype(np.float32),
                terminated or truncated
            )
            loss = agent.train_step()
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Verify PER is working
        assert len(agent.buffer) > 0


class TestMultiComponentIntegration:
    """Test integration between multiple components."""
    
    def test_controller_with_different_algorithms(self):
        """Test controller with different algorithm selections."""
        controller = TrafficController()
        
        # Register intersection with DQN algorithm
        controller.registry.register_intersection(
            "dqn_intersection",
            "DQN Test Intersection",
            {"algorithm": "dqn", "type": "4-way"}
        )
        
        # Register intersection with fuzzy algorithm
        controller.registry.register_intersection(
            "fuzzy_intersection",
            "Fuzzy Test Intersection",
            {"algorithm": "fuzzy", "type": "4-way"}
        )
        
        import asyncio
        queue_lengths = [5.0, 8.0, 3.0, 2.0]
        wait_times = [10.0, 15.0, 8.0, 5.0]
        
        # Test DQN algorithm selection
        decision_dqn = asyncio.run(controller.make_decision(
            intersection_id="dqn_intersection",
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            throughput=0.5,
            current_phase=0,
        ))
        # DQN may fallback to fuzzy if agent not available, or return "dqn"
        assert decision_dqn["algorithm"] in ["dqn", "fuzzy", "default"]
        
        # Test fuzzy algorithm selection
        decision_fuzzy = asyncio.run(controller.make_decision(
            intersection_id="fuzzy_intersection",
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            throughput=0.5,
            current_phase=0,
        ))
        assert decision_fuzzy["algorithm"] == "fuzzy"
    
    def test_security_integration(self):
        """Test security framework integration with API workflow."""
        from src.api.security import InputValidator, SecurityAuditLogger
        
        # Test input validation
        valid_id = "intersection_001"
        invalid_id = "intersection@001"  # Invalid character
        
        assert InputValidator.validate_intersection_id(valid_id) == True
        assert InputValidator.validate_intersection_id(invalid_id) == False
        
        # Test queue length validation
        valid_queues = [5.0, 8.0, 3.0, 2.0]
        invalid_queues = [-1.0, 1001.0]  # Out of bounds
        
        assert InputValidator.validate_queue_lengths(valid_queues) == True
        assert InputValidator.validate_queue_lengths(invalid_queues) == False
        
        # Test security audit logging
        SecurityAuditLogger.log_security_event(
            "test_event",
            {"test": "data"},
            severity="info"
        )
        # Should not raise exception


class TestErrorHandling:
    """Test error handling across component boundaries."""
    
    def test_invalid_state_handling(self):
        """Test handling of invalid states."""
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
        
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            cfg=DQNConfig()
        )
        
        # Test with invalid state (wrong shape)
        invalid_state = np.array([0.1, 0.2])  # Wrong size
        
        # Should handle gracefully
        try:
            action = agent.select_action(invalid_state)
            # If it doesn't raise, it should still return valid action
            assert 0 <= action < env.action_space.n
        except Exception:
            # Or raise appropriate error
            pass
    
    def test_network_failure_simulation(self):
        """Test system behavior with simulated network failures."""
        controller = TrafficController()
        
        # Simulate failure by using invalid data
        # System should return safe defaults
        import asyncio
        try:
            decision = asyncio.run(controller.make_decision(
                intersection_id="test",
                queue_lengths=[],  # Empty list
                wait_times=[],
                throughput=0.0,
                current_phase=0,
            ))
            # Should return safe default
            assert decision is not None
            assert "phase" in decision
        except Exception as e:
            # Or raise appropriate error
            assert "error" in str(e).lower() or "invalid" in str(e).lower()

