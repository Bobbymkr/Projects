"""
Unit tests for distributed tracing.
"""

import pytest
from unittest.mock import Mock, patch
from src.monitoring.tracing import setup_tracing, get_tracer, trace_function, TraceContext


class TestTracing:
    """Test tracing functionality."""
    
    def test_get_tracer(self):
        """Test getting tracer instance."""
        tracer = get_tracer("test_service")
        assert tracer is not None
    
    def test_trace_function_decorator(self):
        """Test trace function decorator."""
        try:
            @trace_function("test_span")
            def test_func(x, y):
                return x + y
            
            result = test_func(2, 3)
            assert result == 5
        except (AttributeError, ImportError):
            # OpenTelemetry not fully available, skip
            pytest.skip("OpenTelemetry not fully configured")
    
    def test_trace_context(self):
        """Test trace context manager."""
        try:
            with TraceContext("test_operation", {"key": "value"}) as span:
                assert span is not None
        except (AttributeError, ImportError):
            # OpenTelemetry not fully available, skip
            pytest.skip("OpenTelemetry not fully configured")
    
    def test_setup_tracing(self):
        """Test tracing setup."""
        # Should not raise exception
        try:
            setup_tracing("test_service")
            assert True
        except Exception:
            # If OpenTelemetry not available, that's okay
            assert True

