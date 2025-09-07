import pytest
import traci
from src.env.marl_env import MarlEnv as MARLEnvironment

@pytest.fixture
def sumo_env():
    # Fixture to set up and tear down SUMO environment
    traci.start(["sumo", "-c", "configs/grid.sumocfg"])
    yield
    traci.close()

def test_environment_initialization(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg")
    assert env is not None
    assert env.num_agents >= 0  # Check that environment has agents

def test_reset(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg")
    observations = env.reset()
    assert isinstance(observations, list)
    assert len(observations) == env.num_agents

def test_step(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg")
    env.reset()
    actions = [0] * env.num_agents  # Sample actions for all agents
    observations, rewards, dones, infos = env.step(actions)
    assert isinstance(observations, list)
    assert isinstance(rewards, list)
    assert isinstance(dones, list)
    assert isinstance(infos, list)
    assert len(observations) == env.num_agents
