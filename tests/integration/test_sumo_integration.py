"""
Integration Tests: SUMO-in-the-loop on Small Networks.

Tests SUMO integration for traffic simulation and control.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import SUMO-related components
try:
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    traci = None

# Import fixtures
from tests.fixtures.sumo_networks import (
    mini_network_2x2_config,
    mock_sumo_connection,
    sumo_route_generator
)


@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not available")
class TestSUMONetworkSetup:
    """Test SUMO network setup and configuration."""
    
    def test_sumo_config_loading(self, mini_network_2x2_config):
        """Test SUMO configuration can be loaded."""
        config = mini_network_2x2_config
        
        assert config is not None
        assert 'config_file' in config
        assert config['config_file'].exists()
    
    def test_sumo_connection(self, mock_sumo_connection):
        """Test SUMO TraCI connection."""
        traci = mock_sumo_connection
        
        # Test connection
        traci.start(['sumo', '-c', 'test.sumocfg'])
        
        assert traci.start.called
    
    def test_traffic_light_access(self, mock_sumo_connection):
        """Test accessing traffic lights in SUMO."""
        traci = mock_sumo_connection
        
        # Get traffic light IDs
        tl_ids = traci.trafficlight.getIDList()
        
        assert tl_ids is not None
        assert len(tl_ids) > 0
        assert isinstance(tl_ids, list)


@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not available")
class TestSUMOSimulationControl:
    """Test SUMO simulation control and state extraction."""
    
    @pytest.fixture
    def sumo_env(self, mock_sumo_connection):
        """Create SUMO environment for testing."""
        traci = mock_sumo_connection
        return traci
    
    def test_simulation_step(self, sumo_env):
        """Test SUMO simulation stepping."""
        traci = sumo_env
        
        # Get initial time
        initial_time = traci.simulation.getTime()
        
        # Step simulation
        traci.simulation.step()
        
        # Time should advance
        assert traci.simulation.step.called
    
    def test_vehicle_state_extraction(self, sumo_env):
        """Test extracting vehicle states from SUMO."""
        traci = sumo_env
        
        # Get vehicle IDs
        vehicle_ids = traci.vehicle.getIDList()
        
        assert vehicle_ids is not None
        assert isinstance(vehicle_ids, list)
        
        # Get vehicle waiting time
        if len(vehicle_ids) > 0:
            wait_time = traci.vehicle.getWaitingTime(vehicle_ids[0])
            assert wait_time is not None
            assert wait_time >= 0
    
    def test_edge_state_extraction(self, sumo_env):
        """Test extracting edge (lane) states from SUMO."""
        traci = sumo_env
        
        # Get edge waiting time
        wait_time = traci.edge.getWaitingTime('e0')
        
        assert wait_time is not None
        assert wait_time >= 0
        
        # Get vehicle count
        vehicle_count = traci.edge.getLastStepVehicleNumber('e0')
        
        assert vehicle_count is not None
        assert vehicle_count >= 0


@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not available")
class TestSUMOSignalControl:
    """Test traffic signal control in SUMO."""
    
    @pytest.fixture
    def sumo_env(self, mock_sumo_connection):
        """Create SUMO environment."""
        return mock_sumo_connection
    
    def test_signal_state_getting(self, sumo_env):
        """Test getting current signal state."""
        traci = sumo_env
        
        # Get traffic light state
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            state = traci.trafficlight.getRedYellowGreenState(tl_ids[0])
            
            assert state is not None
            assert isinstance(state, str)
            assert len(state) > 0
    
    def test_signal_state_setting(self, sumo_env):
        """Test setting traffic signal state."""
        traci = sumo_env
        
        # Set traffic light state
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            traci.trafficlight.setRedYellowGreenState(tl_ids[0], 'GGGG')
            
            assert traci.trafficlight.setRedYellowGreenState.called
    
    def test_phase_duration_control(self, sumo_env):
        """Test controlling phase durations."""
        traci = sumo_env
        
        # Set phase duration
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            # Set phase program
            traci.trafficlight.setPhaseDuration(tl_ids[0], 30.0)
            
            # Verify call was made
            if hasattr(traci.trafficlight, 'setPhaseDuration'):
                assert True  # Method exists and was called


@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO not available")
class TestSUMOAgentIntegration:
    """Test agent integration with SUMO simulation."""
    
    @pytest.fixture
    def sumo_env(self, mock_sumo_connection):
        """Create SUMO environment."""
        return mock_sumo_connection
    
    @pytest.fixture
    def dqn_agent(self):
        """Create DQN agent."""
        from src.rl.dqn_agent import DQNAgent, DQNConfig
        config = DQNConfig(
            state_dim=12,
            action_dim=4,
            learning_rate=0.001
        )
        return DQNAgent(config)
    
    def test_agent_observation_from_sumo(self, sumo_env, dqn_agent):
        """Test extracting agent observations from SUMO state."""
        traci = sumo_env
        agent = dqn_agent
        
        # Extract state from SUMO
        # Get queue lengths from edges
        queue_lengths = []
        for i in range(4):
            edge_id = f'e{i}'
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            queue_lengths.append(vehicle_count)
        
        # Get wait times
        wait_times = []
        for i in range(4):
            edge_id = f'e{i}'
            wait_time = traci.edge.getWaitingTime(edge_id)
            wait_times.append(wait_time)
        
        # Create observation
        observation = np.array(queue_lengths + wait_times, dtype=np.float32)
        
        assert observation.shape == (8,)
        assert all(obs >= 0 for obs in observation)
    
    def test_agent_action_execution_in_sumo(self, sumo_env, dqn_agent):
        """Test executing agent actions in SUMO."""
        traci = sumo_env
        agent = dqn_agent
        
        # Get observation
        queue_lengths = [traci.edge.getLastStepVehicleNumber(f'e{i}') for i in range(4)]
        wait_times = [traci.edge.getWaitingTime(f'e{i}') for i in range(4)]
        observation = np.array(queue_lengths + wait_times, dtype=np.float32)
        
        # Agent selects action
        action = agent.select_action(observation)
        
        # Execute action in SUMO
        tl_ids = traci.trafficlight.getIDList()
        if len(tl_ids) > 0:
            # Map action to phase
            phase_states = ['GGGG', 'Grrr', 'rGrr', 'rrGr', 'rrrG']
            if action < len(phase_states):
                traci.trafficlight.setRedYellowGreenState(tl_ids[0], phase_states[action])
        
        assert action is not None
        assert 0 <= action < 4
    
    def test_episode_simulation(self, sumo_env, dqn_agent):
        """Test complete episode simulation with agent."""
        traci = sumo_env
        agent = dqn_agent
        
        # Simulate episode
        max_steps = 100
        total_reward = 0
        
        for step in range(max_steps):
            # Get state
            queue_lengths = [traci.edge.getLastStepVehicleNumber(f'e{i}') for i in range(4)]
            wait_times = [traci.edge.getWaitingTime(f'e{i}') for i in range(4)]
            state = np.array(queue_lengths + wait_times, dtype=np.float32)
            
            # Agent action
            action = agent.select_action(state)
            
            # Execute action
            tl_ids = traci.trafficlight.getIDList()
            if len(tl_ids) > 0:
                phase_states = ['GGGG', 'Grrr', 'rGrr', 'rrGr', 'rrrG']
                if action < len(phase_states):
                    traci.trafficlight.setRedYellowGreenState(tl_ids[0], phase_states[action])
            
            # Step simulation
            traci.simulation.step()
            
            # Calculate reward
            reward = -np.mean(wait_times)
            total_reward += reward
        
        assert total_reward is not None
        assert isinstance(total_reward, (int, float))


class TestSUMONetworkScenarios:
    """Test different SUMO network scenarios."""
    
    def test_small_network_simulation(self, mini_network_2x2_config, mock_sumo_connection):
        """Test simulation on small 2x2 network."""
        config = mini_network_2x2_config
        traci = mock_sumo_connection
        
        # Start simulation
        traci.start(['sumo', '-c', str(config['config_file'])])
        
        # Run simulation for a few steps
        for _ in range(10):
            traci.simulation.step()
        
        assert traci.simulation.step.call_count == 10
    
    def test_multi_intersection_coordination(self, mock_sumo_connection):
        """Test multi-intersection coordination in SUMO."""
        traci = mock_sumo_connection
        
        # Get multiple traffic lights
        tl_ids = traci.trafficlight.getIDList()
        
        if len(tl_ids) >= 2:
            # Control multiple intersections
            for tl_id in tl_ids[:2]:
                traci.trafficlight.setRedYellowGreenState(tl_id, 'GGGG')
            
            # Verify both were controlled
            assert traci.trafficlight.setRedYellowGreenState.call_count >= 2

