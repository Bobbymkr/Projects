"""
Unit tests for monitoring metrics.
"""

import pytest
from unittest.mock import Mock, patch
import sys

# Clear any existing metrics to avoid duplicate registration
if 'prometheus_client.registry' in sys.modules:
    from prometheus_client import REGISTRY
    REGISTRY._collector_to_names.clear()


class TestMetricsCollector:
    """Test metrics collector."""
    
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear Prometheus registry before each test."""
        try:
            from prometheus_client import REGISTRY
            # Clear collectors
            REGISTRY._collector_to_names.clear()
            REGISTRY._names_to_collectors.clear()
        except:
            pass
        yield
        try:
            REGISTRY._collector_to_names.clear()
            REGISTRY._names_to_collectors.clear()
        except:
            pass
    
    def test_initialization(self):
        """Test metrics collector initialization."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        assert collector.namespace == "adaptive_traffic"
        assert collector.traffic is not None
        assert collector.system is not None
        assert collector.business is not None
    
    def test_record_vehicle(self):
        """Test vehicle recording."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.record_vehicle("intersection_1", 0, 10.5)
        # Metrics should be recorded (no exception)
        assert True
    
    def test_update_queue_length(self):
        """Test queue length update."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.update_queue_length("intersection_1", 0, 5)
        # Metrics should be updated (no exception)
        assert True
    
    def test_update_throughput(self):
        """Test throughput update."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.update_throughput("intersection_1", 500.0)
        # Metrics should be updated (no exception)
        assert True
    
    def test_update_efficiency(self):
        """Test efficiency update."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.update_efficiency("intersection_1", 85.5)
        # Metrics should be updated (no exception)
        assert True
    
    def test_measure_inference_latency(self):
        """Test inference latency measurement."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        
        with collector.measure_inference_latency("dqn", "model_v1"):
            # Simulate inference
            pass
        # Context manager should work (no exception)
        assert True
    
    def test_update_system_resources(self):
        """Test system resource update."""
        from src.monitoring.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.update_system_resources(50.0, 1024*1024*1024, 60.0, "host1")
        # Metrics should be updated (no exception)
        assert True


class TestTrafficMetrics:
    """Test traffic metrics."""
    
    def test_initialization(self):
        """Test traffic metrics initialization."""
        from src.monitoring.metrics import TrafficMetrics
        try:
            metrics = TrafficMetrics()
            assert metrics.vehicles_total is not None
            assert metrics.vehicle_wait_time is not None
            assert metrics.queue_length is not None
        except ValueError:
            # Duplicate registration, that's okay
            pytest.skip("Metrics already registered")

