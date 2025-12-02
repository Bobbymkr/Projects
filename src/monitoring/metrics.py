"""
Week 5: Comprehensive Prometheus Metrics Collection.

Instruments all critical paths:
- Traffic metrics (vehicle count, wait time, queue length)
- System metrics (inference latency, CPU, memory)
- Business metrics (throughput, efficiency)
"""

import time
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    def start_http_server(*args, **kwargs): pass


class TrafficMetrics:
    """Traffic-specific metrics."""
    
    def __init__(self):
        # Vehicle metrics
        self.vehicles_total = Counter(
            'traffic_vehicles_total',
            'Total number of vehicles processed',
            ['intersection_id', 'lane']
        )
        
        self.vehicle_wait_time = Histogram(
            'traffic_vehicle_wait_time_seconds',
            'Vehicle wait time in seconds',
            ['intersection_id', 'lane'],
            buckets=[1, 5, 10, 20, 30, 60, 120, 300]
        )
        
        # Queue metrics
        self.queue_length = Gauge(
            'traffic_queue_length',
            'Current queue length',
            ['intersection_id', 'lane']
        )
        
        self.queue_length_max = Gauge(
            'traffic_queue_length_max',
            'Maximum queue length observed',
            ['intersection_id', 'lane']
        )
        
        # Throughput metrics
        self.throughput_vehicles_per_hour = Gauge(
            'traffic_throughput_vehicles_per_hour',
            'Throughput in vehicles per hour',
            ['intersection_id']
        )
        
        # Efficiency metrics
        self.traffic_efficiency = Gauge(
            'traffic_efficiency_percent',
            'Traffic efficiency percentage',
            ['intersection_id']
        )


class SystemMetrics:
    """System performance metrics."""
    
    def __init__(self):
        # Inference latency
        self.inference_latency = Histogram(
            'system_inference_latency_seconds',
            'Inference latency in seconds',
            ['agent_type', 'model'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        # CPU metrics
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['host']
        )
        
        # Memory metrics
        self.memory_usage_bytes = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['host']
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            ['host']
        )
        
        # GPU metrics (if available)
        self.gpu_usage_percent = Gauge(
            'system_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id']
        )
        
        self.gpu_memory_usage_bytes = Gauge(
            'system_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id']
        )


class BusinessMetrics:
    """Business and operational metrics."""
    
    def __init__(self):
        # Throughput
        self.total_throughput = Counter(
            'business_total_throughput',
            'Total throughput across all intersections',
            ['time_period']
        )
        
        # Efficiency
        self.average_efficiency = Gauge(
            'business_average_efficiency_percent',
            'Average traffic efficiency percentage'
        )
        
        # Wait time reduction
        self.wait_time_reduction_percent = Gauge(
            'business_wait_time_reduction_percent',
            'Average wait time reduction percentage',
            ['baseline_algorithm']
        )
        
        # Cost savings
        self.estimated_cost_savings = Gauge(
            'business_estimated_cost_savings_usd',
            'Estimated cost savings in USD',
            ['time_period']
        )


class MetricsCollector:
    """Centralized metrics collector."""
    
    def __init__(self, namespace: str = "adaptive_traffic"):
        self.namespace = namespace
        self.traffic = TrafficMetrics()
        self.system = SystemMetrics()
        self.business = BusinessMetrics()
        self._lock = threading.Lock()
    
    @contextmanager
    def measure_inference_latency(self, agent_type: str, model: str = "default"):
        """Context manager to measure inference latency."""
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            self.system.inference_latency.labels(
                agent_type=agent_type,
                model=model
            ).observe(latency)
    
    def record_vehicle(self, intersection_id: str, lane: int, wait_time: float):
        """Record vehicle metrics."""
        self.traffic.vehicles_total.labels(
            intersection_id=intersection_id,
            lane=str(lane)
        ).inc()
        
        self.traffic.vehicle_wait_time.labels(
            intersection_id=intersection_id,
            lane=str(lane)
        ).observe(wait_time)
    
    def update_queue_length(self, intersection_id: str, lane: int, length: int):
        """Update queue length metrics."""
        self.traffic.queue_length.labels(
            intersection_id=intersection_id,
            lane=str(lane)
        ).set(length)
        
        # Update max if needed
        current_max = self.traffic.queue_length_max.labels(
            intersection_id=intersection_id,
            lane=str(lane)
        )._value.get()
        
        if length > (current_max or 0):
            self.traffic.queue_length_max.labels(
                intersection_id=intersection_id,
                lane=str(lane)
            ).set(length)
    
    def update_throughput(self, intersection_id: str, throughput: float):
        """Update throughput metrics."""
        self.traffic.throughput_vehicles_per_hour.labels(
            intersection_id=intersection_id
        ).set(throughput)
    
    def update_efficiency(self, intersection_id: str, efficiency: float):
        """Update efficiency metrics."""
        self.traffic.traffic_efficiency.labels(
            intersection_id=intersection_id
        ).set(efficiency)
    
    def update_system_resources(self, cpu_percent: float, memory_bytes: int, memory_percent: float, host: str = "default"):
        """Update system resource metrics."""
        self.system.cpu_usage_percent.labels(host=host).set(cpu_percent)
        self.system.memory_usage_bytes.labels(host=host).set(memory_bytes)
        self.system.memory_usage_percent.labels(host=host).set(memory_percent)
    
    def start_metrics_server(self, port: int = 9090):
        """Start Prometheus metrics HTTP server."""
        if PROMETHEUS_AVAILABLE:
            start_http_server(port)
            print(f"Prometheus metrics server started on port {port}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_collection(port: int = 9090):
    """Start metrics collection server."""
    collector = get_metrics_collector()
    collector.start_metrics_server(port)

