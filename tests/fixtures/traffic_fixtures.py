"""
Traffic Control Test Fixtures.

Reusable fixtures for testing traffic control functionality.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


@pytest.fixture
def sample_intersection_id() -> str:
    """Sample intersection ID."""
    return "intersection_1"


@pytest.fixture
def sample_traffic_state() -> Dict[str, Any]:
    """Sample traffic state for testing."""
    return {
        "queue_lengths": [5, 3, 8, 2],
        "wait_times": [12.5, 8.3, 15.2, 6.1],
        "arrival_rates": [0.3, 0.2, 0.4, 0.15],
        "current_phase": 0,
        "phase_start_time": 0.0,
    }


@pytest.fixture
def sample_traffic_state_high_load() -> Dict[str, Any]:
    """Sample traffic state with high load."""
    return {
        "queue_lengths": [25, 30, 28, 22],
        "wait_times": [45.2, 52.1, 48.7, 38.9],
        "arrival_rates": [0.8, 0.9, 0.85, 0.75],
        "current_phase": 0,
        "phase_start_time": 0.0,
    }


@pytest.fixture
def sample_traffic_state_low_load() -> Dict[str, Any]:
    """Sample traffic state with low load."""
    return {
        "queue_lengths": [1, 0, 2, 1],
        "wait_times": [3.2, 1.5, 4.1, 2.8],
        "arrival_rates": [0.1, 0.05, 0.12, 0.08],
        "current_phase": 0,
        "phase_start_time": 0.0,
    }


@pytest.fixture
def sample_traffic_state_imbalanced() -> Dict[str, Any]:
    """Sample traffic state with imbalanced lanes."""
    return {
        "queue_lengths": [20, 2, 1, 1],
        "wait_times": [50.0, 5.0, 3.0, 2.0],
        "arrival_rates": [0.9, 0.1, 0.05, 0.05],
        "current_phase": 0,
        "phase_start_time": 0.0,
    }


@pytest.fixture
def mock_dqn_agent_config() -> Dict[str, Any]:
    """Mock DQN agent configuration."""
    return {
        "state_dim": 12,
        "action_dim": 4,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "batch_size": 32,
        "memory_size": 10000,
    }


@pytest.fixture
def mock_intersection_config() -> Dict[str, Any]:
    """Mock intersection configuration."""
    return {
        "id": "intersection_1",
        "name": "Test Intersection",
        "phases": 4,
        "min_phase_duration": 15,
        "max_phase_duration": 120,
        "yellow_duration": 3,
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
        },
    }


@pytest.fixture
def sample_decision_request() -> Dict[str, Any]:
    """Sample traffic decision request."""
    return {
        "intersection_id": "intersection_1",
        "current_state": {
            "queue_lengths": [5, 3, 8, 2],
            "wait_times": [12.5, 8.3, 15.2, 6.1],
            "arrival_rates": [0.3, 0.2, 0.4, 0.15],
        },
        "algorithm": "dqn",
        "metadata": {
            "timestamp": "2024-11-30T12:00:00Z",
            "episode": 100,
        },
    }


@pytest.fixture
def sample_decision_response() -> Dict[str, Any]:
    """Sample traffic decision response."""
    return {
        "decision": {
            "phase": 0,
            "duration": 30,
            "confidence": 0.92,
        },
        "performance": {
            "expected_wait_reduction": 0.35,
            "queue_reduction": 0.28,
        },
        "metadata": {
            "algorithm": "dqn",
            "processing_time_ms": 45,
            "model_version": "v2.1.0",
        },
    }


class TrafficStateGenerator:
    """Generator for test traffic states."""
    
    @staticmethod
    def generate_random_state(
        num_lanes: int = 4,
        queue_range: tuple = (0, 30),
        wait_range: tuple = (0, 60),
        arrival_range: tuple = (0.0, 1.0),
    ) -> Dict[str, Any]:
        """Generate random traffic state."""
        return {
            "queue_lengths": list(np.random.randint(queue_range[0], queue_range[1], num_lanes)),
            "wait_times": list(np.random.uniform(wait_range[0], wait_range[1], num_lanes)),
            "arrival_rates": list(np.random.uniform(arrival_range[0], arrival_range[1], num_lanes)),
            "current_phase": np.random.randint(0, num_lanes),
            "phase_start_time": np.random.uniform(0, 120),
        }
    
    @staticmethod
    def generate_sequence(
        num_states: int = 10,
        num_lanes: int = 4,
    ) -> List[Dict[str, Any]]:
        """Generate sequence of traffic states."""
        states = []
        current_state = TrafficStateGenerator.generate_random_state(num_lanes)
        
        for _ in range(num_states):
            states.append(current_state.copy())
            # Evolve state
            for i in range(num_lanes):
                # Simulate queue growth
                current_state["queue_lengths"][i] += np.random.randint(-2, 5)
                current_state["queue_lengths"][i] = max(0, current_state["queue_lengths"][i])
                
                # Update wait times
                current_state["wait_times"][i] = (
                    current_state["queue_lengths"][i] * 2.5 + np.random.uniform(-1, 1)
                )
        
        return states


@pytest.fixture
def traffic_state_generator() -> TrafficStateGenerator:
    """Traffic state generator fixture."""
    return TrafficStateGenerator()


@pytest.fixture
def sample_traffic_sequence(traffic_state_generator) -> List[Dict[str, Any]]:
    """Sample sequence of traffic states."""
    return traffic_state_generator.generate_sequence(num_states=10)


# Environment simulators

@pytest.fixture
def mock_traffic_environment():
    """Mock traffic environment for testing."""
    from unittest.mock import Mock
    
    env = Mock()
    env.reset.return_value = np.random.rand(12)
    env.step.return_value = (
        np.random.rand(12),  # next_state
        10.5,  # reward
        False,  # done
        {},  # info
    )
    env.action_space.n = 4
    env.observation_space.shape = (12,)
    return env

