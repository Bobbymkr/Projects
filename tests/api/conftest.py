"""
Pytest Configuration and Shared Fixtures for API Tests.

Provides common fixtures, test clients, and test configuration
for all API test suites.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Generator
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.api.config import settings
from src.api.monitoring import setup_prometheus_metrics


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    # Override settings for test environment
    settings.ENVIRONMENT = "test"
    settings.DEBUG = True
    settings.ENABLE_SENTRY = False
    settings.ENABLE_REDIS = False
    settings.ENABLE_PROMETHEUS = True
    settings.ENABLE_CORS = True
    return settings


@pytest.fixture(scope="session")
def metrics():
    """Create metrics instance for testing."""
    return setup_prometheus_metrics()


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.
    
    Yields:
        TestClient instance for making test requests
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_traffic_controller(monkeypatch):
    """Mock traffic controller for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    mock_controller = AsyncMock()
    mock_controller.make_decision = AsyncMock(return_value={
        "phase": 1,
        "green_time": 25.0,
        "confidence": 0.85,
        "algorithm": "fuzzy",
        "reasoning": "Test decision",
        "improvement": 15.5,
    })
    mock_controller.get_all_intersections = AsyncMock(return_value=[])
    mock_controller.get_intersection = AsyncMock(return_value=None)
    
    # Patch the dependency
    from src.api.dependencies import get_traffic_controller
    monkeypatch.setattr("src.api.dependencies.get_traffic_controller", lambda: mock_controller)
    
    return mock_controller


@pytest.fixture
def sample_traffic_decision_request():
    """Sample traffic decision request data."""
    return {
        "intersection_id": "int-001",
        "queue_lengths": [12.5, 8.3, 15.2, 10.1],
        "wait_times": [25.3, 18.7, 32.1, 22.5],
        "throughput": 450.0,
        "current_phase": 1,
    }


@pytest.fixture
def sample_traffic_decision_response():
    """Sample traffic decision response data."""
    return {
        "intersection_id": "int-001",
        "recommended_phase": 1,
        "green_time": 25.0,
        "confidence": 0.85,
        "algorithm_used": "fuzzy",
        "reasoning": "Test decision",
        "estimated_improvement": 15.5,
        "processing_time_ms": 12.5,
        "timestamp": "2025-11-30T12:00:00Z",
    }


@pytest.fixture
def sample_batch_request():
    """Sample batch traffic decision request."""
    return {
        "requests": [
            {
                "intersection_id": "int-001",
                "queue_lengths": [12, 8, 15, 10],
                "wait_times": [25, 18, 32, 22],
                "throughput": 450,
                "current_phase": 1,
            },
            {
                "intersection_id": "int-002",
                "queue_lengths": [10, 12, 9, 11],
                "wait_times": [20, 25, 18, 22],
                "throughput": 380,
                "current_phase": 2,
            },
        ]
    }

