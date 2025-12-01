"""
Unit Tests for System Management API Routes.

Tests for health checks, system status, and system information endpoints.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestHealthCheckEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(
        self,
        client: TestClient,
    ):
        """Test root health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert data["status"] == "healthy"
    
    def test_system_health(
        self,
        client: TestClient,
    ):
        """Test detailed system health endpoint."""
        response = client.get("/api/v1/system/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "active_intersections" in data
        assert "total_requests" in data
        assert "error_rate" in data
        assert "timestamp" in data
        
        # Validate ranges
        assert 0.0 <= data["cpu_usage_percent"] <= 100.0
        assert 0.0 <= data["memory_usage_percent"] <= 100.0
        assert data["active_intersections"] >= 0
        assert 0.0 <= data["error_rate"] <= 1.0


class TestSystemStatusEndpoint:
    """Tests for system status endpoint."""
    
    def test_system_status(
        self,
        client: TestClient,
    ):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data


class TestSystemInfoEndpoint:
    """Tests for system information endpoint."""
    
    def test_system_info(
        self,
        client: TestClient,
    ):
        """Test system information endpoint."""
        response = client.get("/api/v1/system/info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "capabilities" in data
        assert "algorithms" in data
        assert "timestamp" in data
        
        assert isinstance(data["capabilities"], list)
        assert isinstance(data["algorithms"], list)
        assert len(data["capabilities"]) > 0
        assert len(data["algorithms"]) > 0

