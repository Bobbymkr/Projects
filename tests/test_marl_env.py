import pytest
import traci
from src.rl.marl_env import MARLEnvironment

@pytest.fixture
def sumo_env():
    # Fixture to set up and tear down SUMO environment
    traci.start(["sumo", "-c", "configs/grid.sumocfg"])
    yield
    traci.close()

def test_environment_initialization(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg", use_gui=False, traffic_lights=["0"])
    assert env is not None
    assert len(env.agents) == 1  # Assuming one traffic light for test

def test_reset(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg", use_gui=False, traffic_lights=["0"])
    observations = env.reset()
    assert isinstance(observations, dict)
    assert "0" in observations

def test_step(sumo_env):
    env = MARLEnvironment(config_path="configs/grid.sumocfg", use_gui=False, traffic_lights=["0"])
    env.reset()
    actions = {"0": 0}  # Sample action
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)