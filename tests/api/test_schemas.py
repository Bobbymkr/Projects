"""
Unit Tests for Pydantic Schemas.

Tests for request/response validation and data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.api.schemas import (
    TrafficDecisionRequest,
    TrafficDecisionResponse,
    BatchTrafficDecisionRequest,
    SystemHealthResponse,
    KPIMetricsResponse,
)


class TestTrafficDecisionRequest:
    """Tests for TrafficDecisionRequest schema."""
    
    def test_valid_request(
        self,
    ):
        """Test valid traffic decision request."""
        request = TrafficDecisionRequest(
            intersection_id="int-001",
            queue_lengths=[12.5, 8.3, 15.2, 10.1],
            wait_times=[25.3, 18.7, 32.1, 22.5],
            throughput=450.0,
            current_phase=1,
        )
        
        assert request.intersection_id == "int-001"
        assert len(request.queue_lengths) == 4
        assert len(request.wait_times) == 4
        assert request.throughput == 450.0
        assert request.current_phase == 1
    
    def test_invalid_phase_range(
        self,
    ):
        """Test invalid phase value."""
        with pytest.raises(ValidationError):
            TrafficDecisionRequest(
                intersection_id="int-001",
                queue_lengths=[12, 8, 15, 10],
                wait_times=[25, 18, 32, 22],
                throughput=450,
                current_phase=5,  # Invalid: should be 0-3
            )
    
    def test_empty_queue_lengths(
        self,
    ):
        """Test empty queue lengths."""
        with pytest.raises(ValidationError):
            TrafficDecisionRequest(
                intersection_id="int-001",
                queue_lengths=[],  # Empty
                wait_times=[25, 18, 32, 22],
                throughput=450,
                current_phase=1,
            )
    
    def test_mismatched_lengths(
        self,
    ):
        """Test mismatched queue and wait times lengths."""
        with pytest.raises(ValidationError):
            TrafficDecisionRequest(
                intersection_id="int-001",
                queue_lengths=[12, 8, 15],
                wait_times=[25, 18, 32, 22],  # Different length
                throughput=450,
                current_phase=1,
            )


class TestTrafficDecisionResponse:
    """Tests for TrafficDecisionResponse schema."""
    
    def test_valid_response(
        self,
    ):
        """Test valid traffic decision response."""
        response = TrafficDecisionResponse(
            intersection_id="int-001",
            recommended_phase=1,
            green_time=25.0,
            confidence=0.85,
            algorithm_used="fuzzy",
            reasoning="Test decision",
            estimated_improvement=15.5,
            processing_time_ms=12.5,
            timestamp=datetime.utcnow(),
        )
        
        assert response.intersection_id == "int-001"
        assert response.recommended_phase == 1
        assert response.confidence == 0.85
        assert response.algorithm_used == "fuzzy"


class TestBatchTrafficDecisionRequest:
    """Tests for BatchTrafficDecisionRequest schema."""
    
    def test_valid_batch_request(
        self,
    ):
        """Test valid batch request."""
        batch_request = BatchTrafficDecisionRequest(
            requests=[
                TrafficDecisionRequest(
                    intersection_id="int-001",
                    queue_lengths=[12, 8, 15, 10],
                    wait_times=[25, 18, 32, 22],
                    throughput=450,
                    current_phase=1,
                ),
                TrafficDecisionRequest(
                    intersection_id="int-002",
                    queue_lengths=[10, 12, 9, 11],
                    wait_times=[20, 25, 18, 22],
                    throughput=380,
                    current_phase=2,
                ),
            ]
        )
        
        assert len(batch_request.requests) == 2
    
    def test_empty_batch_request(
        self,
    ):
        """Test empty batch request."""
        with pytest.raises(ValidationError):
            BatchTrafficDecisionRequest(requests=[])


class TestSystemHealthResponse:
    """Tests for SystemHealthResponse schema."""
    
    def test_valid_system_health(
        self,
    ):
        """Test valid system health response."""
        health = SystemHealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            cpu_usage_percent=45.5,
            memory_usage_percent=52.3,
            active_intersections=4,
            total_requests=1000,
            error_rate=0.02,
            timestamp=datetime.utcnow(),
        )
        
        assert health.status == "healthy"
        assert 0.0 <= health.cpu_usage_percent <= 100.0
        assert 0.0 <= health.memory_usage_percent <= 100.0
        assert 0.0 <= health.error_rate <= 1.0

