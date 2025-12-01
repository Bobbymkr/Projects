"""
Unit Tests for Traffic Control API Routes.

Tests for traffic decision endpoints, batch processing, and intersection management.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestTrafficDecisionEndpoint:
    """Tests for single traffic decision endpoint."""
    
    def test_make_traffic_decision_success(
        self,
        client: TestClient,
        mock_traffic_controller,
        sample_traffic_decision_request,
    ):
        """Test successful traffic decision request."""
        response = client.post(
            "/api/v1/traffic/decision",
            json=sample_traffic_decision_request,
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "intersection_id" in data
        assert "recommended_phase" in data
        assert "green_time" in data
        assert "confidence" in data
        assert "algorithm_used" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data
        
        assert data["intersection_id"] == sample_traffic_decision_request["intersection_id"]
        assert 0 <= data["recommended_phase"] <= 3
        assert 5.0 <= data["green_time"] <= 60.0
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_make_traffic_decision_invalid_request(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test traffic decision with invalid request data."""
        # Missing required field
        invalid_request = sample_traffic_decision_request.copy()
        del invalid_request["intersection_id"]
        
        response = client.post(
            "/api/v1/traffic/decision",
            json=invalid_request,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_make_traffic_decision_invalid_queue_lengths(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test traffic decision with invalid queue lengths."""
        invalid_request = sample_traffic_decision_request.copy()
        invalid_request["queue_lengths"] = []  # Empty list
        
        response = client.post(
            "/api/v1/traffic/decision",
            json=invalid_request,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_make_traffic_decision_mismatched_lengths(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test traffic decision with mismatched array lengths."""
        invalid_request = sample_traffic_decision_request.copy()
        invalid_request["wait_times"] = [25.3, 18.7]  # Different length
        
        response = client.post(
            "/api/v1/traffic/decision",
            json=invalid_request,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_make_traffic_decision_invalid_phase(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test traffic decision with invalid phase value."""
        invalid_request = sample_traffic_decision_request.copy()
        invalid_request["current_phase"] = 5  # Invalid phase
        
        response = client.post(
            "/api/v1/traffic/decision",
            json=invalid_request,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBatchTrafficDecisionEndpoint:
    """Tests for batch traffic decision endpoint."""
    
    def test_batch_decision_success(
        self,
        client: TestClient,
        mock_traffic_controller,
        sample_batch_request,
    ):
        """Test successful batch traffic decision request."""
        response = client.post(
            "/api/v1/traffic/batch",
            json=sample_batch_request,
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "decisions" in data
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "total_processing_time_ms" in data
        assert "timestamp" in data
        
        assert data["total_requests"] == len(sample_batch_request["requests"])
        assert len(data["decisions"]) == data["successful_requests"]
        assert data["total_processing_time_ms"] > 0
    
    def test_batch_decision_empty_list(
        self,
        client: TestClient,
    ):
        """Test batch decision with empty request list."""
        response = client.post(
            "/api/v1/traffic/batch",
            json={"requests": []},
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_decision_too_many_requests(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test batch decision with too many requests."""
        too_many_requests = {
            "requests": [sample_traffic_decision_request] * 101  # Exceeds max
        }
        
        response = client.post(
            "/api/v1/traffic/batch",
            json=too_many_requests,
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestIntersectionsEndpoint:
    """Tests for intersection management endpoints."""
    
    def test_get_intersections(
        self,
        client: TestClient,
        mock_traffic_controller,
    ):
        """Test getting all intersections."""
        response = client.get("/api/v1/traffic/intersections")
        
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
    
    def test_get_intersections_with_status_filter(
        self,
        client: TestClient,
        mock_traffic_controller,
    ):
        """Test getting intersections with status filter."""
        response = client.get("/api/v1/traffic/intersections?status=active")
        
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
    
    def test_get_intersection_by_id(
        self,
        client: TestClient,
        mock_traffic_controller,
    ):
        """Test getting intersection by ID."""
        response = client.get("/api/v1/traffic/intersections/int-001")
        
        # Returns 404 if not found, which is expected with mock
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
        ]
    
    def test_get_intersection_not_found(
        self,
        client: TestClient,
        mock_traffic_controller,
    ):
        """Test getting non-existent intersection."""
        response = client.get("/api/v1/traffic/intersections/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

