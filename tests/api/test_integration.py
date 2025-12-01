"""
Integration Tests for API Endpoints.

Tests that verify the integration between multiple components
and end-to-end workflows.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestTrafficDecisionWorkflow:
    """Integration tests for traffic decision workflow."""
    
    def test_complete_decision_workflow(
        self,
        client: TestClient,
        sample_traffic_decision_request,
    ):
        """Test complete workflow from request to response."""
        # Make decision request
        decision_response = client.post(
            "/api/v1/traffic/decision",
            json=sample_traffic_decision_request,
        )
        
        assert decision_response.status_code == status.HTTP_200_OK
        decision_data = decision_response.json()
        
        # Verify response structure
        assert "recommended_phase" in decision_data
        assert "green_time" in decision_data
        assert "confidence" in decision_data
        
        # Verify metrics are updated
        metrics_response = client.get("/api/v1/metrics/performance")
        assert metrics_response.status_code == status.HTTP_200_OK
    
    def test_batch_decision_workflow(
        self,
        client: TestClient,
        sample_batch_request,
    ):
        """Test batch decision workflow."""
        # Make batch request
        batch_response = client.post(
            "/api/v1/traffic/batch",
            json=sample_batch_request,
        )
        
        assert batch_response.status_code == status.HTTP_200_OK
        batch_data = batch_response.json()
        
        # Verify batch processing results
        assert batch_data["total_requests"] == len(sample_batch_request["requests"])
        assert len(batch_data["decisions"]) <= batch_data["total_requests"]


class TestDashboardIntegration:
    """Integration tests for dashboard data flow."""
    
    def test_dashboard_data_aggregation(
        self,
        client: TestClient,
    ):
        """Test that dashboard endpoint aggregates data correctly."""
        dashboard_response = client.get("/api/v1/metrics/dashboard")
        
        assert dashboard_response.status_code == status.HTTP_200_OK
        dashboard_data = dashboard_response.json()
        
        # Verify all required sections are present
        required_sections = ["kpis", "performance", "system_health", "alerts"]
        for section in required_sections:
            assert section in dashboard_data
        
        # Verify data consistency
        kpis = dashboard_data["kpis"]
        if "active_intersections" in kpis and "total_intersections" in kpis:
            assert kpis["active_intersections"] <= kpis["total_intersections"]


class TestSystemHealthIntegration:
    """Integration tests for system health monitoring."""
    
    def test_health_check_chain(
        self,
        client: TestClient,
    ):
        """Test health check endpoint chain."""
        # Root health check
        root_health = client.get("/health")
        assert root_health.status_code == status.HTTP_200_OK
        
        # System health check
        system_health = client.get("/api/v1/system/health")
        assert system_health.status_code == status.HTTP_200_OK
        
        # System status
        system_status = client.get("/api/v1/system/status")
        assert system_status.status_code == status.HTTP_200_OK
        
        # All should indicate healthy status
        assert root_health.json()["status"] == "healthy"
        assert system_health.json()["status"] in ["healthy", "warning", "critical"]

