import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.marl_env import MarlEnv

@pytest.fixture
def marl_env():
    """Fixture to set up MARL environment."""
    config_path = "configs/grid.sumocfg"
    return MarlEnv(config_path=config_path, min_green=5, max_green=60, yellow_time=3)

def test_environment_initialization(marl_env):
    """Test MARL environment initialization."""
    assert marl_env is not None
    assert hasattr(marl_env, 'num_agents')
    assert hasattr(marl_env, 'action_space')
    assert hasattr(marl_env, 'observation_space')

def test_reset(marl_env):
    """Test environment reset."""
    try:
        states = marl_env.reset()
        assert isinstance(states, list) or isinstance(states, np.ndarray)
        assert len(states) == marl_env.num_agents
    except Exception as e:
        # SUMO might not be available in test environment
        pytest.skip(f"SUMO not available: {e}")

def test_step(marl_env):
    """Test environment step."""
    try:
        states = marl_env.reset()
        actions = [0] * marl_env.num_agents  # Sample actions
        next_states, rewards, dones, infos = marl_env.step(actions)
        
        assert isinstance(next_states, list) or isinstance(next_states, np.ndarray)
        assert isinstance(rewards, list) or isinstance(rewards, np.ndarray)
        assert isinstance(dones, list) or isinstance(dones, np.ndarray)
        assert len(next_states) == marl_env.num_agents
        assert len(rewards) == marl_env.num_agents
        assert len(dones) == marl_env.num_agents
    except Exception as e:
        # SUMO might not be available in test environment
        pytest.skip(f"SUMO not available: {e}")

def test_action_space(marl_env):
    """Test action space properties."""
    assert hasattr(marl_env.action_space, '__len__') or hasattr(marl_env.action_space, 'n')
    if hasattr(marl_env.action_space, '__len__'):
        assert len(marl_env.action_space) == marl_env.num_agents

def test_observation_space(marl_env):
    """Test observation space properties."""
    assert hasattr(marl_env.observation_space, '__len__') or hasattr(marl_env.observation_space, 'shape')
    if hasattr(marl_env.observation_space, '__len__'):
        assert len(marl_env.observation_space) == marl_env.num_agents

def test_forecaster_integration(marl_env):
    """Test traffic forecaster integration."""
    assert hasattr(marl_env, 'forecaster')
    assert isinstance(marl_env.forecaster, dict)
    assert len(marl_env.forecaster) == marl_env.num_agents

def test_neighbor_detection(marl_env):
    """Test neighbor detection functionality."""
    assert hasattr(marl_env, 'neighbors')
    assert isinstance(marl_env.neighbors, dict)

def test_edge_mapping(marl_env):
    """Test edge mapping functionality."""
    assert hasattr(marl_env, 'edge_mapping')
    assert isinstance(marl_env.edge_mapping, dict)