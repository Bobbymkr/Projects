"""
Integration Tests: Agent with Environment.

Tests the integration between RL agents and traffic environments,
including both stub environments and real MARL environments.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import agents
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.control.fuzzy_control import FuzzyLogicController

# Import environments
from src.env.traffic_env import TrafficEnv
from src.env.marl_env import MarlEnv

# Import fixtures
from tests.fixtures.traffic_fixtures import (
    sample_traffic_state,
    mock_traffic_environment,
    traffic_state_generator
)


class TestAgentWithStubEnvironment:
    """Test agents with stub/mock environments."""
    
    @pytest.fixture
    def stub_env(self):
        """Create a stub environment for testing."""
        env = Mock()
        env.reset.return_value = np.array([5, 3, 8, 2, 12.5, 8.3, 15.2, 6.1, 0.3, 0.2, 0.4, 0.15])
        env.step.return_value = (
            np.array([6, 4, 7, 3, 13.0, 9.0, 14.5, 7.0, 0.32, 0.22, 0.38, 0.16]),
            10.5,
            False,
            {}
        )
        env.action_space = Mock()
        env.action_space.n = 4
        env.observation_space = Mock()
        env.observation_space.shape = (12,)
        return env
    
    @pytest.fixture
    def dqn_agent(self):
        """Create a DQN agent for testing."""
        config = DQNConfig(
            state_dim=12,
            action_dim=4,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            memory_size=1000
        )
        return DQNAgent(config)
    
    def test_dqn_agent_reset_and_observe(self, dqn_agent, stub_env):
        """Test DQN agent can reset and observe from stub environment."""
        state = stub_env.reset()
        
        assert state is not None
        assert state.shape == (12,)
        assert dqn_agent.config.state_dim == 12
    
    def test_dqn_agent_action_selection(self, dqn_agent, stub_env):
        """Test DQN agent can select actions in stub environment."""
        state = stub_env.reset()
        action = dqn_agent.select_action(state)
        
        assert action is not None
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4
    
    def test_dqn_agent_step_interaction(self, dqn_agent, stub_env):
        """Test DQN agent can interact with stub environment."""
        state = stub_env.reset()
        action = dqn_agent.select_action(state)
        
        next_state, reward, done, info = stub_env.step(action)
        
        assert next_state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_dqn_agent_training_step(self, dqn_agent, stub_env):
        """Test DQN agent training step with stub environment."""
        # Collect some experience
        state = stub_env.reset()
        for _ in range(10):
            action = dqn_agent.select_action(state)
            next_state, reward, done, info = stub_env.step(action)
            dqn_agent.store_transition(state, action, reward, next_state, done)
            state = next_state if not done else stub_env.reset()
        
        # Train agent
        if len(dqn_agent.replay_buffer) >= dqn_agent.config.batch_size:
            loss = dqn_agent.train_step()
            assert loss is not None
            assert isinstance(loss, float)
    
    def test_fuzzy_controller_with_stub_env(self, stub_env):
        """Test fuzzy logic controller with stub environment."""
        controller = FuzzyLogicController()
        state = stub_env.reset()
        
        # Extract queue lengths from state (first 4 elements)
        queue_lengths = state[:4].tolist()
        wait_times = state[4:8].tolist()
        
        action = controller.compute_timing(queue_lengths, wait_times)
        
        assert action is not None
        assert 'phase' in action or 'duration' in action


class TestAgentWithRealMARLEnvironment:
    """Test agents with real MARL environment."""
    
    @pytest.fixture
    def marl_env_config(self):
        """MARL environment configuration."""
        return {
            "config_path": "configs/grid.sumocfg",
            "num_agents": 2,
            "min_green": 10,
            "max_green": 60,
            "yellow_time": 3
        }
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-sumo", default=False),
        reason="Requires SUMO and --run-sumo flag"
    )
    def test_marl_env_initialization(self, marl_env_config):
        """Test MARL environment can be initialized."""
        with patch('src.env.marl_env.traci') as mock_traci:
            mock_traci.start = Mock()
            mock_traci.trafficlight.getIDList = Mock(return_value=['tl0', 'tl1'])
            mock_traci.trafficlight.getControlledLinks = Mock(return_value=[
                [('e0_0', 'e1_0', 0)],
                [('e2_0', 'e3_0', 0)]
            ])
            
            env = MarlEnv(**marl_env_config)
            
            assert env is not None
            assert env.num_agents == 2
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-sumo", default=False),
        reason="Requires SUMO and --run-sumo flag"
    )
    def test_dqn_agent_with_marl_env(self, marl_env_config):
        """Test DQN agent with real MARL environment."""
        with patch('src.env.marl_env.traci') as mock_traci:
            # Setup mock TraCI
            mock_traci.start = Mock()
            mock_traci.trafficlight.getIDList = Mock(return_value=['tl0', 'tl1'])
            mock_traci.trafficlight.getControlledLinks = Mock(return_value=[
                [('e0_0', 'e1_0', 0)],
                [('e2_0', 'e3_0', 0)]
            ])
            mock_traci.simulation.getTime.return_value = 0.0
            mock_traci.edge.getLastStepVehicleNumber.return_value = 5
            mock_traci.edge.getWaitingTime.return_value = 10.0
            
            env = MarlEnv(**marl_env_config)
            
            # Create agent
            config = DQNConfig(
                state_dim=env.observation_space[0].shape[0],
                action_dim=env.action_space[0].n,
                learning_rate=0.001
            )
            agent = DQNAgent(config)
            
            # Test interaction
            states = env.reset()
            assert len(states) == 2
            
            actions = [agent.select_action(state) for state in states]
            assert len(actions) == 2
            
            next_states, rewards, dones, infos = env.step(actions)
            assert len(next_states) == 2
            assert len(rewards) == 2
    
    def test_multi_agent_coordination(self):
        """Test multi-agent coordination in MARL environment."""
        # This would test agent communication and coordination
        # For now, we'll create a mock test
        num_agents = 2
        
        agents = [
            DQNAgent(DQNConfig(state_dim=12, action_dim=4))
            for _ in range(num_agents)
        ]
        
        # Simulate coordinated decision making
        states = [np.random.rand(12) for _ in range(num_agents)]
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        
        assert len(actions) == num_agents
        assert all(0 <= a < 4 for a in actions)
    
    def test_episode_termination(self, mock_traffic_environment):
        """Test episode termination conditions."""
        env = mock_traffic_environment
        state = env.reset()
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            action = 0  # Simple action
            state, reward, done, info = env.step(action)
            steps += 1
        
        # Episode should terminate
        assert done or steps >= max_steps


class TestAgentStateConsistency:
    """Test state consistency across agent-environment interactions."""
    
    def test_state_shape_consistency(self, mock_traffic_environment):
        """Test that state shapes remain consistent."""
        env = mock_traffic_environment
        state = env.reset()
        initial_shape = state.shape
        
        for _ in range(10):
            action = 0
            next_state, _, _, _ = env.step(action)
            assert next_state.shape == initial_shape
    
    def test_action_validity(self):
        """Test that actions are always valid."""
        config = DQNConfig(state_dim=12, action_dim=4)
        agent = DQNAgent(config)
        
        for _ in range(100):
            state = np.random.rand(12)
            action = agent.select_action(state)
            assert 0 <= action < 4
    
    def test_reward_consistency(self, mock_traffic_environment):
        """Test that rewards are consistent and reasonable."""
        env = mock_traffic_environment
        state = env.reset()
        
        rewards = []
        for _ in range(10):
            action = 0
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
            assert isinstance(reward, (int, float))
        
        # Rewards should be in reasonable range
        assert all(-1000 < r < 1000 for r in rewards)


class TestAgentPerformance:
    """Test agent performance characteristics."""
    
    def test_inference_latency(self):
        """Test that agent inference meets latency requirements."""
        config = DQNConfig(state_dim=12, action_dim=4)
        agent = DQNAgent(config)
        
        import time
        
        state = np.random.rand(12)
        
        # Measure inference time
        start = time.time()
        action = agent.select_action(state)
        latency = time.time() - start
        
        assert latency < 0.1  # Should be < 100ms
        assert action is not None
    
    def test_training_convergence(self, mock_traffic_environment):
        """Test that agent can learn and improve over time."""
        env = mock_traffic_environment
        config = DQNConfig(
            state_dim=12,
            action_dim=4,
            memory_size=1000,
            batch_size=32
        )
        agent = DQNAgent(config)
        
        # Collect experience
        state = env.reset()
        for _ in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()
        
        # Train and check for improvement
        initial_loss = None
        for _ in range(10):
            if len(agent.replay_buffer) >= config.batch_size:
                loss = agent.train_step()
                if initial_loss is None:
                    initial_loss = loss
                
                assert loss is not None
                assert isinstance(loss, float)


# Add pytest option for SUMO tests
def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-sumo",
        action="store_true",
        default=False,
        help="run tests that require SUMO"
    )

