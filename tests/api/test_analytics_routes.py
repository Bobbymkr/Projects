"""
Unit Tests for Analytics API Routes.

Tests for algorithm performance, traffic patterns, and causal analysis endpoints.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestAlgorithmPerformanceEndpoint:
    """Tests for algorithm performance comparison endpoint."""
    
    def test_get_algorithm_performance(
        self,
        client: TestClient,
    ):
        """Test getting algorithm performance comparison."""
        response = client.get("/api/v1/analytics/algorithm-performance")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Validate algorithm data structure
        for algo in data:
            assert "key" in algo
            assert "name" in algo
            assert "wait_time" in algo
            assert "improvement" in algo
            assert "grade" in algo
            
            assert algo["wait_time"] >= 0
            assert algo["improvement"] >= 0


class TestTrafficPatternsEndpoint:
    """Tests for traffic patterns endpoint."""
    
    def test_get_traffic_patterns(
        self,
        client: TestClient,
    ):
        """Test getting 24-hour traffic patterns."""
        response = client.get("/api/v1/analytics/traffic-patterns")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "hours" in data
        assert "volumes" in data
        assert "speeds" in data
        
        assert isinstance(data["hours"], list)
        assert isinstance(data["volumes"], list)
        assert isinstance(data["speeds"], list)
        
        assert len(data["hours"]) == 24
        assert len(data["volumes"]) == 24
        assert len(data["speeds"]) == 24


class TestCausalAnalysisEndpoint:
    """Tests for causal analysis endpoint."""
    
    def test_get_causal_analysis(
        self,
        client: TestClient,
    ):
        """Test getting causal analysis of traffic factors."""
        response = client.get("/api/v1/analytics/causal-analysis")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Validate causal factor structure
        for factor in data:
            assert "factor" in factor
            assert "impact" in factor
            assert "confidence" in factor
            
            assert 0.0 <= factor["impact"] <= 1.0
            assert 0.0 <= factor["confidence"] <= 1.0

