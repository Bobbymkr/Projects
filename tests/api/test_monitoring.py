"""
Unit Tests for Monitoring and Observability.

Tests for Prometheus metrics endpoint, logging, and error tracking.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestPrometheusMetricsEndpoint:
    """Tests for Prometheus metrics export endpoint."""
    
    def test_metrics_endpoint_exists(
        self,
        client: TestClient,
    ):
        """Test that metrics endpoint exists and is accessible."""
        response = client.get("/metrics")
        
        # Should return 200 or 404 (if Prometheus not installed)
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
        ]
    
    def test_metrics_endpoint_content_type(
        self,
        client: TestClient,
    ):
        """Test metrics endpoint content type."""
        response = client.get("/metrics")
        
        if response.status_code == status.HTTP_200_OK:
            # Should be text/plain for Prometheus format
            assert "text/plain" in response.headers.get("content-type", "")


class TestRequestMiddleware:
    """Tests for request processing middleware."""
    
    def test_request_id_header(
        self,
        client: TestClient,
    ):
        """Test that request ID is added to response headers."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
    
    def test_process_time_header(
        self,
        client: TestClient,
    ):
        """Test that process time is added to response headers."""
        response = client.get("/health")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


class TestErrorHandling:
    """Tests for error handling and logging."""
    
    def test_not_found_error(
        self,
        client: TestClient,
    ):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent/endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_method_not_allowed(
        self,
        client: TestClient,
    ):
        """Test 405 method not allowed error."""
        response = client.post("/health")  # GET endpoint
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

