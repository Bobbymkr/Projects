"""
Unit Tests for Metrics and Analytics API Routes.

Tests for KPI metrics, performance metrics, and dashboard endpoints.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestKPIMetricsEndpoint:
    """Tests for KPI metrics endpoint."""
    
    def test_get_kpi_metrics(
        self,
        client: TestClient,
    ):
        """Test getting KPI metrics."""
        response = client.get("/api/v1/metrics/kpis")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_intersections" in data
        assert "active_intersections" in data
        assert "total_vehicles_processed" in data
        assert "average_response_time_ms" in data
        assert "cost_savings" in data
        assert "environmental_impact" in data
        assert "user_satisfaction" in data
        assert "system_efficiency" in data
        assert "timestamp" in data
        
        # Validate ranges
        assert data["total_intersections"] >= 0
        assert data["active_intersections"] >= 0
        assert data["total_vehicles_processed"] >= 0
        assert 0.0 <= data["user_satisfaction"] <= 5.0
        assert 0.0 <= data["system_efficiency"] <= 100.0
    
    def test_get_kpi_metrics_with_time_range(
        self,
        client: TestClient,
    ):
        """Test getting KPI metrics with time range filter."""
        response = client.get("/api/v1/metrics/kpis?time_range=week")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_intersections" in data


class TestPerformanceMetricsEndpoint:
    """Tests for performance metrics endpoint."""
    
    def test_get_performance_metrics(
        self,
        client: TestClient,
    ):
        """Test getting performance metrics."""
        response = client.get("/api/v1/metrics/performance")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "endpoint" in data
        assert "requests_total" in data
        assert "requests_per_second" in data
        assert "average_response_time_ms" in data
        assert "p50_response_time_ms" in data
        assert "p95_response_time_ms" in data
        assert "p99_response_time_ms" in data
        assert "error_rate" in data
        assert "success_rate" in data
        
        # Validate ranges
        assert 0.0 <= data["error_rate"] <= 1.0
        assert 0.0 <= data["success_rate"] <= 1.0
        assert data["p50_response_time_ms"] <= data["p95_response_time_ms"]
        assert data["p95_response_time_ms"] <= data["p99_response_time_ms"]
    
    def test_get_performance_metrics_with_endpoint_filter(
        self,
        client: TestClient,
    ):
        """Test getting performance metrics for specific endpoint."""
        response = client.get("/api/v1/metrics/performance?endpoint=/api/v1/traffic/decision")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["endpoint"] == "/api/v1/traffic/decision"


class TestDashboardMetricsEndpoint:
    """Tests for dashboard metrics endpoint."""
    
    def test_get_dashboard_metrics(
        self,
        client: TestClient,
    ):
        """Test getting aggregated dashboard metrics."""
        response = client.get("/api/v1/metrics/dashboard")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "kpis" in data
        assert "performance" in data
        assert "system_health" in data
        assert "alerts" in data
        assert "timestamp" in data
        
        # Validate KPIs structure
        kpis = data["kpis"]
        assert "total_intersections" in kpis
        assert "active_intersections" in kpis
        assert "environmental_impact" in kpis
        
        # Validate performance structure
        performance = data["performance"]
        assert "system_efficiency" in performance
        
        # Validate system health structure
        system_health = data["system_health"]
        assert "status" in system_health
        assert "cpu_usage" in system_health

