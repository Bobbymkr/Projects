"""
Prometheus Metrics Setup.

Comprehensive metrics collection for monitoring and observability.
"""

try:
    from prometheus_client import Counter, Histogram, Gauge
except ImportError:
    # Fallback if prometheus_client not installed
    class Counter:
        def __init__(self, *args, **kwargs):
            self._value = type('obj', (object,), {'get': lambda: 0})()
        def labels(self, **kwargs):
            return self
        def inc(self, value=1):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass

import logging

logger = logging.getLogger(__name__)


def setup_prometheus_metrics():
    """
    Setup Prometheus metrics for the application.
    
    Returns:
        Metrics object with all registered metrics
    """
    
    class Metrics:
        """Container for all Prometheus metrics."""
        
        # HTTP metrics
        http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        
        http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        
        # Traffic decision metrics
        traffic_decision_requests_total = Counter(
            "traffic_decision_requests_total",
            "Total traffic decision requests",
            ["algorithm"],
        )
        
        traffic_decision_duration_seconds = Histogram(
            "traffic_decision_duration_seconds",
            "Traffic decision processing time",
            ["algorithm"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )
        
        traffic_decision_errors_total = Counter(
            "traffic_decision_errors_total",
            "Total traffic decision errors",
            ["error_type"],
        )
        
        # System metrics
        active_websocket_connections = Gauge(
            "active_websocket_connections",
            "Number of active WebSocket connections",
        )
        
        system_cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
        )
        
        system_memory_usage = Gauge(
            "system_memory_usage_percent",
            "System memory usage percentage",
        )
        
        # Business metrics
        total_intersections = Gauge(
            "total_intersections",
            "Total number of intersections in the system",
        )
        
        active_intersections = Gauge(
            "active_intersections",
            "Number of active intersections",
        )
        
        vehicles_processed_total = Counter(
            "vehicles_processed_total",
            "Total vehicles processed",
            ["intersection_id"],
        )
        
        average_wait_time_seconds = Histogram(
            "average_wait_time_seconds",
            "Average vehicle wait time in seconds",
            ["intersection_id"],
            buckets=[5, 10, 15, 20, 30, 45, 60, 90, 120],
        )
        
        queue_length = Gauge(
            "queue_length",
            "Current queue length",
            ["intersection_id", "lane_id"],
        )
        
        # API performance metrics
        api_response_time_seconds = Histogram(
            "api_response_time_seconds",
            "API endpoint response time",
            ["method", "endpoint", "status_code"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        
        api_errors_total = Counter(
            "api_errors_total",
            "Total API errors",
            ["method", "endpoint", "error_type"],
        )
        
        # WebSocket metrics
        websocket_messages_sent_total = Counter(
            "websocket_messages_sent_total",
            "Total WebSocket messages sent",
            ["channel"],
        )
        
        websocket_messages_received_total = Counter(
            "websocket_messages_received_total",
            "Total WebSocket messages received",
            ["channel"],
        )
    
    return Metrics()
