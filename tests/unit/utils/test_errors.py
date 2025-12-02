"""
Unit tests for error handling utilities.
"""

import pytest
from src.utils.errors import (
    AdaptiveTrafficError,
    ConfigurationError,
    ResourceError,
    ModelError
)


class TestErrors:
    """Test error classes."""
    
    def test_adaptive_traffic_error(self):
        """Test AdaptiveTrafficError."""
        error = AdaptiveTrafficError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Config failed")
        assert str(error) == "Config failed"
        assert isinstance(error, AdaptiveTrafficError)
    
    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError("Resource unavailable")
        assert str(error) == "Resource unavailable"
        assert isinstance(error, AdaptiveTrafficError)
    
    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Model issue")
        assert str(error) == "Model issue"
        assert isinstance(error, AdaptiveTrafficError)

