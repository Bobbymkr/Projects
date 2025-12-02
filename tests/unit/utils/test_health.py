"""
Unit tests for health check utilities.
"""

import pytest
from src.utils.health import HealthCheck


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_initialization(self):
        """Test health check initialization."""
        health = HealthCheck()
        assert health is not None
        assert health.quota_manager is None
        assert isinstance(health.components, dict)
    
    def test_register_component(self):
        """Test component registration."""
        health = HealthCheck()
        
        def check_func():
            return True, {"status": "ok"}
        
        health.register_component("test_component", check_func)
        assert "test_component" in health.components
    
    def test_check_system(self):
        """Test system check."""
        health = HealthCheck()
        status = health.check_system()
        assert status is not None
        assert isinstance(status, dict)
        assert "status" in status
        assert "details" in status
    
    def test_check_quota(self):
        """Test quota check without quota manager."""
        health = HealthCheck()
        status = health.check_quota()
        assert status is not None
        assert isinstance(status, dict)
        assert status["status"] == "unknown"
    
    def test_check_all(self):
        """Test checking all components."""
        health = HealthCheck()
        results = health.check_all()
        assert results is not None
        assert isinstance(results, dict)
        # Results may have "system" or "components" key
        assert "system" in results or "components" in results or "status" in results
