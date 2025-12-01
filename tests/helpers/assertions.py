"""
Custom Assertion Utilities.

Extended assertion functions for traffic control testing.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def assert_valid_phase(phase: int, num_phases: int = 4):
    """Assert phase is valid."""
    assert isinstance(phase, (int, np.integer)), f"Phase must be integer, got {type(phase)}"
    assert 0 <= phase < num_phases, f"Phase must be in [0, {num_phases}), got {phase}"


def assert_valid_duration(duration: float, min_duration: float = 15.0, max_duration: float = 120.0):
    """Assert duration is valid."""
    assert isinstance(duration, (int, float, np.number)), f"Duration must be numeric, got {type(duration)}"
    assert min_duration <= duration <= max_duration, \
        f"Duration must be in [{min_duration}, {max_duration}], got {duration}"


def assert_valid_queue_lengths(queue_lengths: List[int], num_lanes: int = 4):
    """Assert queue lengths are valid."""
    assert isinstance(queue_lengths, list), "queue_lengths must be a list"
    assert len(queue_lengths) == num_lanes, \
        f"queue_lengths must have {num_lanes} elements, got {len(queue_lengths)}"
    assert all(isinstance(q, (int, np.integer)) for q in queue_lengths), \
        "All queue lengths must be integers"
    assert all(q >= 0 for q in queue_lengths), "All queue lengths must be non-negative"


def assert_valid_wait_times(wait_times: List[float], num_lanes: int = 4):
    """Assert wait times are valid."""
    assert isinstance(wait_times, list), "wait_times must be a list"
    assert len(wait_times) == num_lanes, \
        f"wait_times must have {num_lanes} elements, got {len(wait_times)}"
    assert all(isinstance(w, (int, float, np.number)) for w in wait_times), \
        "All wait times must be numeric"
    assert all(w >= 0 for w in wait_times), "All wait times must be non-negative"


def assert_valid_decision(decision: Dict[str, Any]):
    """Assert decision structure is valid."""
    assert isinstance(decision, dict), "Decision must be a dictionary"
    assert "phase" in decision, "Decision must contain 'phase'"
    assert "duration" in decision, "Decision must contain 'duration'"
    
    assert_valid_phase(decision["phase"])
    assert_valid_duration(decision["duration"])
    
    if "confidence" in decision:
        confidence = decision["confidence"]
        assert 0 <= confidence <= 1, f"Confidence must be in [0, 1], got {confidence}"


def assert_valid_traffic_state(state: Dict[str, Any]):
    """Assert traffic state structure is valid."""
    assert isinstance(state, dict), "State must be a dictionary"
    
    if "queue_lengths" in state:
        assert_valid_queue_lengths(state["queue_lengths"])
    
    if "wait_times" in state:
        num_lanes = len(state.get("queue_lengths", [4]))
        assert_valid_wait_times(state["wait_times"], num_lanes)
    
    if "arrival_rates" in state:
        arrival_rates = state["arrival_rates"]
        assert isinstance(arrival_rates, list), "arrival_rates must be a list"
        assert all(0 <= r <= 2.0 for r in arrival_rates), \
            "Arrival rates should be reasonable (0-2.0 vehicles/second)"


def assert_performance_improvement(
    baseline_metric: float,
    improved_metric: float,
    min_improvement: float = 0.1,
):
    """Assert performance improvement meets minimum threshold."""
    improvement = (baseline_metric - improved_metric) / baseline_metric
    assert improvement >= min_improvement, \
        f"Expected improvement >= {min_improvement}, got {improvement:.2%}"


def assert_response_time(response_time_ms: float, max_time_ms: float = 100.0):
    """Assert API response time is acceptable."""
    assert response_time_ms < max_time_ms, \
        f"Response time {response_time_ms}ms exceeds maximum {max_time_ms}ms"


def assert_decision_latency(latency_ms: float, max_latency_ms: float = 50.0):
    """Assert decision-making latency is acceptable."""
    assert latency_ms < max_latency_ms, \
        f"Decision latency {latency_ms}ms exceeds maximum {max_latency_ms}ms"

