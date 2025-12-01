"""
Testing Utilities and Helpers.

Reusable testing fixtures, mocks, and helpers for comprehensive testing.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class MockTrafficState:
    """Mock traffic state for testing."""
    queue_lengths: np.ndarray
    wait_times: np.ndarray
    vehicle_counts: np.ndarray
    intersection_id: str = "test_intersection"
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_lengths": self.queue_lengths.tolist(),
            "wait_times": self.wait_times.tolist(),
            "vehicle_counts": self.vehicle_counts.tolist(),
            "intersection_id": self.intersection_id,
            "timestamp": self.timestamp,
        }


class TrafficDataGenerator:
    """
    Generator for mock traffic data for testing.
    
    Creates realistic traffic scenarios for algorithm testing.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def generate_state(
        self,
        num_lanes: int = 4,
        intersection_id: str = "test",
        **kwargs: Any,
    ) -> MockTrafficState:
        """
        Generate a random traffic state.
        
        Args:
            num_lanes: Number of traffic lanes
            intersection_id: Intersection identifier
            **kwargs: Additional state parameters
            
        Returns:
            Mock traffic state
        """
        queue_lengths = np.random.randint(0, 20, size=num_lanes)
        wait_times = np.random.uniform(0, 30, size=num_lanes)
        vehicle_counts = np.random.randint(0, 50, size=num_lanes)
        
        return MockTrafficState(
            queue_lengths=queue_lengths,
            wait_times=wait_times,
            vehicle_counts=vehicle_counts,
            intersection_id=intersection_id,
            timestamp=kwargs.get("timestamp", 0.0),
        )
    
    def generate_state_sequence(
        self,
        num_steps: int = 100,
        num_lanes: int = 4,
        intersection_id: str = "test",
    ) -> List[MockTrafficState]:
        """
        Generate a sequence of traffic states.
        
        Args:
            num_steps: Number of time steps
            num_lanes: Number of traffic lanes
            intersection_id: Intersection identifier
            
        Returns:
            List of traffic states
        """
        states = []
        for step in range(num_steps):
            state = self.generate_state(
                num_lanes=num_lanes,
                intersection_id=intersection_id,
                timestamp=float(step),
            )
            states.append(state)
        
        return states
    
    def generate_rush_hour_scenario(
        self,
        duration: int = 60,
        peak_start: int = 20,
        peak_end: int = 40,
    ) -> List[MockTrafficState]:
        """
        Generate rush hour traffic scenario.
        
        Args:
            duration: Total duration in minutes
            peak_start: Peak traffic start time
            peak_end: Peak traffic end time
            
        Returns:
            List of traffic states representing rush hour
        """
        states = []
        for minute in range(duration):
            if peak_start <= minute <= peak_end:
                # High traffic during peak
                intensity = 0.8 + 0.2 * np.random.random()
            else:
                # Low traffic otherwise
                intensity = 0.2 + 0.3 * np.random.random()
            
            queue_lengths = (intensity * 30 * np.random.random(4)).astype(int)
            wait_times = intensity * 40 * np.random.random(4)
            vehicle_counts = (intensity * 60 * np.random.random(4)).astype(int)
            
            state = MockTrafficState(
                queue_lengths=queue_lengths,
                wait_times=wait_times,
                vehicle_counts=vehicle_counts,
                timestamp=float(minute),
            )
            states.append(state)
        
        return states


class MockAlgorithm:
    """
    Mock algorithm for testing.
    
    Provides predictable behavior for testing purposes.
    """
    
    def __init__(self, action_sequence: Optional[List[int]] = None):
        """
        Initialize mock algorithm.
        
        Args:
            action_sequence: Optional sequence of actions to return
        """
        self.action_sequence = action_sequence or []
        self.action_index = 0
        self.call_count = 0
    
    def predict(self, state: Any) -> int:
        """
        Predict action for state.
        
        Args:
            state: Traffic state
            
        Returns:
            Action index
        """
        self.call_count += 1
        
        if self.action_sequence:
            action = self.action_sequence[self.action_index % len(self.action_sequence)]
            self.action_index += 1
            return action
        
        return 0  # Default action
    
    def train(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Mock training method."""
        return {"loss": 0.1, "epoch": 1}


class TestEnvironment:
    """
    Mock environment for testing algorithms.
    
    Simulates traffic environment interactions.
    """
    
    def __init__(
        self,
        num_lanes: int = 4,
        num_actions: int = 4,
        max_steps: int = 1000,
    ):
        """
        Initialize test environment.
        
        Args:
            num_lanes: Number of traffic lanes
            num_actions: Number of possible actions
            max_steps: Maximum steps per episode
        """
        self.num_lanes = num_lanes
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.current_step = 0
        self.state_generator = TrafficDataGenerator(seed=42)
    
    def reset(self) -> MockTrafficState:
        """Reset environment."""
        self.current_step = 0
        return self.state_generator.generate_state(num_lanes=self.num_lanes)
    
    def step(self, action: int) -> tuple[MockTrafficState, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state.
        
        Args:
            action: Action to execute
            
        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Generate next state
        next_state = self.state_generator.generate_state(
            num_lanes=self.num_lanes,
            timestamp=float(self.current_step),
        )
        
        # Calculate reward (simplified)
        reward = -np.mean(next_state.wait_times) / 10.0
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        info = {
            "step": self.current_step,
            "action": action,
        }
        
        return next_state, reward, done, info


class AssertionHelpers:
    """Helper functions for test assertions."""
    
    @staticmethod
    def assert_valid_action(action: int, num_actions: int) -> None:
        """Assert that action is valid."""
        assert 0 <= action < num_actions, f"Invalid action: {action}"
    
    @staticmethod
    def assert_valid_state(state: MockTrafficState) -> None:
        """Assert that state is valid."""
        assert len(state.queue_lengths) > 0, "Empty queue lengths"
        assert len(state.wait_times) > 0, "Empty wait times"
        assert np.all(state.queue_lengths >= 0), "Negative queue lengths"
        assert np.all(state.wait_times >= 0), "Negative wait times"
    
    @staticmethod
    def assert_performance_improvement(
        baseline_metric: float,
        improved_metric: float,
        min_improvement: float = 0.1,
    ) -> None:
        """Assert that performance improved by at least min_improvement."""
        improvement = (baseline_metric - improved_metric) / baseline_metric
        assert improvement >= min_improvement, (
            f"Performance improvement {improvement:.2%} is less than "
            f"minimum {min_improvement:.2%}"
        )


# Convenience fixtures for pytest
def pytest_fixtures():
    """Provide pytest fixture generators."""
    return {
        "traffic_state": lambda: TrafficDataGenerator(seed=42).generate_state(),
        "traffic_scenario": lambda: TrafficDataGenerator(seed=42).generate_rush_hour_scenario(),
        "test_env": lambda: TestEnvironment(),
        "mock_algorithm": lambda: MockAlgorithm(),
    }

