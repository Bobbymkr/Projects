"""Metrics and observability utilities.

This module provides utilities for collecting, reporting, and visualizing
metrics for the adaptive traffic control system, including Prometheus integration,
OpenTelemetry tracing, and dashboard generation.
"""

import time
import logging
import threading
import functools
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from prometheus_client import push_to_gateway, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class MetricsError(Exception):
    """Exception raised for metrics-related errors."""
    pass


class MetricsRegistry:
    """Registry for managing metrics.
    
    This class provides a centralized registry for creating and accessing
    metrics, with support for Prometheus and other backends.
    """
    
    def __init__(self, namespace: str = "adaptive_traffic"):
        """Initialize the metrics registry.
        
        Args:
            namespace: Namespace for metrics
        """
        self.namespace = namespace
        self._metrics = {}
        self._lock = threading.RLock()
        
        # Check if Prometheus is available
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics will be collected but not exported.")
    
    def counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Any:
        """Create or get a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            
        Returns:
            Counter metric
        """
        return self._get_or_create_metric('counter', name, description, labels)
    
    def gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Any:
        """Create or get a gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            
        Returns:
            Gauge metric
        """
        return self._get_or_create_metric('gauge', name, description, labels)
    
    def histogram(self, name: str, description: str, labels: Optional[List[str]] = None, 
                 buckets: Optional[List[float]] = None) -> Any:
        """Create or get a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: List of bucket boundaries
            
        Returns:
            Histogram metric
        """
        return self._get_or_create_metric('histogram', name, description, labels, buckets=buckets)
    
    def summary(self, name: str, description: str, labels: Optional[List[str]] = None) -> Any:
        """Create or get a summary metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            
        Returns:
            Summary metric
        """
        return self._get_or_create_metric('summary', name, description, labels)
    
    def _get_or_create_metric(self, metric_type: str, name: str, description: str, 
                             labels: Optional[List[str]] = None, **kwargs) -> Any:
        """Get or create a metric.
        
        Args:
            metric_type: Type of metric ('counter', 'gauge', 'histogram', 'summary')
            name: Metric name
            description: Metric description
            labels: List of label names
            **kwargs: Additional arguments for the metric
            
        Returns:
            Metric object
        """
        key = f"{metric_type}:{name}"
        
        with self._lock:
            if key in self._metrics:
                return self._metrics[key]
            
            if not PROMETHEUS_AVAILABLE:
                # Create a dummy metric if Prometheus is not available
                metric = DummyMetric(name, description, labels)
            else:
                # Create a Prometheus metric
                if metric_type == 'counter':
                    metric = Counter(name, description, labels or [], namespace=self.namespace)
                elif metric_type == 'gauge':
                    metric = Gauge(name, description, labels or [], namespace=self.namespace)
                elif metric_type == 'histogram':
                    buckets = kwargs.get('buckets')
                    metric = Histogram(name, description, labels or [], buckets=buckets, namespace=self.namespace)
                elif metric_type == 'summary':
                    metric = Summary(name, description, labels or [], namespace=self.namespace)
                else:
                    raise MetricsError(f"Unknown metric type: {metric_type}")
            
            self._metrics[key] = metric
            return metric
    
    def start_http_server(self, port: int = 8000, addr: str = '') -> None:
        """Start a Prometheus HTTP server to expose metrics.
        
        Args:
            port: HTTP port
            addr: Bind address
            
        Raises:
            MetricsError: If Prometheus is not available
        """
        if not PROMETHEUS_AVAILABLE:
            raise MetricsError("Prometheus client not available")
        
        try:
            start_http_server(port, addr)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            raise MetricsError(f"Failed to start Prometheus HTTP server: {str(e)}") from e
    
    def push_to_gateway(self, gateway: str, job: str, registry=None, grouping_key: Optional[Dict[str, str]] = None) -> None:
        """Push metrics to a Prometheus Pushgateway.
        
        Args:
            gateway: Pushgateway address (host:port)
            job: Job name
            registry: Registry to push
            grouping_key: Grouping key for the metrics
            
        Raises:
            MetricsError: If Prometheus is not available
        """
        if not PROMETHEUS_AVAILABLE:
            raise MetricsError("Prometheus client not available")
        
        try:
            push_to_gateway(gateway, job, registry, grouping_key)
            logger.debug(f"Pushed metrics to Pushgateway {gateway} for job {job}")
        except Exception as e:
            logger.warning(f"Failed to push metrics to Pushgateway: {str(e)}")


class DummyMetric:
    """Dummy metric for use when Prometheus is not available."""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize the dummy metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values = {}
    
    def inc(self, amount: float = 1, **labels) -> None:
        """Increment the metric.
        
        Args:
            amount: Amount to increment
            **labels: Label values
        """
        key = self._get_key(labels)
        self._values[key] = self._values.get(key, 0) + amount
    
    def dec(self, amount: float = 1, **labels) -> None:
        """Decrement the metric.
        
        Args:
            amount: Amount to decrement
            **labels: Label values
        """
        key = self._get_key(labels)
        self._values[key] = self._values.get(key, 0) - amount
    
    def set(self, value: float, **labels) -> None:
        """Set the metric value.
        
        Args:
            value: Value to set
            **labels: Label values
        """
        key = self._get_key(labels)
        self._values[key] = value
    
    def observe(self, value: float, **labels) -> None:
        """Observe a value.
        
        Args:
            value: Value to observe
            **labels: Label values
        """
        key = self._get_key(labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)
    
    def _get_key(self, labels: Dict[str, str]) -> str:
        """Get a key for the labels.
        
        Args:
            labels: Label values
            
        Returns:
            Key string
        """
        if not labels:
            return 'default'
        
        parts = []
        for label in self.labels:
            if label in labels:
                parts.append(f"{label}={labels[label]}")
        
        return ','.join(parts) if parts else 'default'


class TracingManager:
    """Manager for distributed tracing.
    
    This class provides utilities for setting up and using OpenTelemetry
    distributed tracing.
    """
    
    def __init__(self, service_name: str = "adaptive_traffic"):
        """Initialize the tracing manager.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self._tracer = None
        
        # Check if OpenTelemetry is available
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Tracing will be disabled.")
    
    def setup_tracing(self, otlp_endpoint: Optional[str] = None) -> None:
        """Set up OpenTelemetry tracing.
        
        Args:
            otlp_endpoint: OTLP exporter endpoint (e.g., 'localhost:4317')
            
        Raises:
            MetricsError: If OpenTelemetry is not available
        """
        if not OPENTELEMETRY_AVAILABLE:
            raise MetricsError("OpenTelemetry not available")
        
        try:
            # Set up the tracer provider
            resource = Resource.create({"service.name": self.service_name})
            tracer_provider = TracerProvider(resource=resource)
            
            # Set up the exporter
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Set the global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            # Get a tracer
            self._tracer = trace.get_tracer(self.service_name)
            
            logger.info(f"Set up OpenTelemetry tracing for service {self.service_name}")
        except Exception as e:
            raise MetricsError(f"Failed to set up OpenTelemetry tracing: {str(e)}") from e
    
    def get_tracer(self) -> Any:
        """Get the OpenTelemetry tracer.
        
        Returns:
            OpenTelemetry tracer
            
        Raises:
            MetricsError: If tracing is not set up
        """
        if not OPENTELEMETRY_AVAILABLE or self._tracer is None:
            raise MetricsError("OpenTelemetry tracing not set up")
        
        return self._tracer
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, str]] = None) -> Any:
        """Create a tracing span.
        
        Args:
            name: Span name
            attributes: Span attributes
            
        Yields:
            OpenTelemetry span
        """
        if not OPENTELEMETRY_AVAILABLE or self._tracer is None:
            # Yield a dummy span if tracing is not available
            yield DummySpan(name)
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    
    def trace(self, name: Optional[str] = None, attributes: Optional[Dict[str, str]] = None):
        """Decorator for tracing functions.
        
        Args:
            name: Span name (defaults to function name)
            attributes: Span attributes
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__name__
                with self.span(span_name, attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        if isinstance(span, Span):  # Check if it's a real span
                            span.set_status(StatusCode.ERROR)
                            span.record_exception(e)
                        raise
            return wrapper
        return decorator


class DummySpan:
    """Dummy span for use when OpenTelemetry is not available."""
    
    def __init__(self, name: str):
        """Initialize the dummy span.
        
        Args:
            name: Span name
        """
        self.name = name
        self.start_time = time.time()
        self.attributes = {}
    
    def set_attribute(self, key: str, value: str) -> None:
        """Set a span attribute.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value
    
    def set_status(self, status: Any) -> None:
        """Set the span status.
        
        Args:
            status: Status code
        """
        pass
    
    def record_exception(self, exception: Exception) -> None:
        """Record an exception.
        
        Args:
            exception: Exception to record
        """
        pass
    
    def end(self) -> None:
        """End the span."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class SystemMetricsCollector:
    """Collector for system metrics.
    
    This class provides utilities for collecting system metrics such as
    CPU usage, memory usage, disk usage, and GPU metrics.
    """
    
    def __init__(self, metrics_registry: Optional[MetricsRegistry] = None):
        """Initialize the system metrics collector.
        
        Args:
            metrics_registry: Metrics registry to use
        """
        self.metrics_registry = metrics_registry or MetricsRegistry()
        
        # Create metrics
        self.cpu_usage = self.metrics_registry.gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = self.metrics_registry.gauge('memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = self.metrics_registry.gauge('disk_usage_percent', 'Disk usage percentage', ['path'])
        
        # GPU metrics if available
        if GPUTIL_AVAILABLE:
            self.gpu_usage = self.metrics_registry.gauge('gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
            self.gpu_memory_usage = self.metrics_registry.gauge('gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id'])
        
        # Check if psutil is available
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available. System metrics collection will be limited.")
    
    def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics.
        
        Returns:
            Dictionary of CPU metrics
        """
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.set(cpu_percent)
            return {'cpu_usage_percent': cpu_percent}
        except Exception as e:
            logger.warning(f"Failed to collect CPU metrics: {str(e)}")
            return {}
    
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics.
        
        Returns:
            Dictionary of memory metrics
        """
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage.set(memory_percent)
            return {'memory_usage_percent': memory_percent}
        except Exception as e:
            logger.warning(f"Failed to collect memory metrics: {str(e)}")
            return {}
    
    def collect_disk_metrics(self, paths: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Collect disk metrics.
        
        Args:
            paths: List of paths to check
            
        Returns:
            Dictionary of disk metrics by path
        """
        if not PSUTIL_AVAILABLE:
            return {}
        
        if paths is None:
            paths = ['/']
        
        result = {}
        for path in paths:
            try:
                disk_usage = psutil.disk_usage(path)
                disk_percent = disk_usage.percent
                self.disk_usage.labels(path=path).set(disk_percent)
                result[path] = {'disk_usage_percent': disk_percent}
            except Exception as e:
                logger.warning(f"Failed to collect disk metrics for {path}: {str(e)}")
        
        return result
    
    def collect_gpu_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect GPU metrics.
        
        Returns:
            Dictionary of GPU metrics by GPU ID
        """
        if not GPUTIL_AVAILABLE:
            return {}
        
        try:
            gpus = GPUtil.getGPUs()
            result = {}
            
            for gpu in gpus:
                gpu_id = str(gpu.id)
                gpu_load = gpu.load * 100  # Convert to percentage
                gpu_memory_percent = gpu.memoryUtil * 100  # Convert to percentage
                
                self.gpu_usage.labels(gpu_id=gpu_id).set(gpu_load)
                self.gpu_memory_usage.labels(gpu_id=gpu_id).set(gpu_memory_percent)
                
                result[gpu_id] = {
                    'gpu_usage_percent': gpu_load,
                    'gpu_memory_usage_percent': gpu_memory_percent
                }
            
            return result
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {str(e)}")
            return {}
    
    def collect_all_metrics(self, disk_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect all system metrics.
        
        Args:
            disk_paths: List of paths to check for disk usage
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Collect CPU metrics
        cpu_metrics = self.collect_cpu_metrics()
        if cpu_metrics:
            metrics['cpu'] = cpu_metrics
        
        # Collect memory metrics
        memory_metrics = self.collect_memory_metrics()
        if memory_metrics:
            metrics['memory'] = memory_metrics
        
        # Collect disk metrics
        disk_metrics = self.collect_disk_metrics(disk_paths)
        if disk_metrics:
            metrics['disk'] = disk_metrics
        
        # Collect GPU metrics
        gpu_metrics = self.collect_gpu_metrics()
        if gpu_metrics:
            metrics['gpu'] = gpu_metrics
        
        return metrics


class ApplicationMetrics:
    """Metrics for the adaptive traffic control application.
    
    This class provides specific metrics for the adaptive traffic control
    application, such as request counts, queue lengths, and RL metrics.
    """
    
    def __init__(self, metrics_registry: Optional[MetricsRegistry] = None):
        """Initialize the application metrics.
        
        Args:
            metrics_registry: Metrics registry to use
        """
        self.metrics_registry = metrics_registry or MetricsRegistry()
        
        # Request metrics
        self.request_count = self.metrics_registry.counter('request_count', 'Number of requests', ['endpoint'])
        self.request_latency = self.metrics_registry.histogram(
            'request_latency_seconds', 'Request latency in seconds', 
            ['endpoint'], buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
        )
        
        # Quota metrics
        self.quota_remaining = self.metrics_registry.gauge('quota_remaining', 'Remaining quota', ['key'])
        
        # RL metrics
        self.episode_reward = self.metrics_registry.gauge('episode_reward', 'Episode reward')
        self.episode_length = self.metrics_registry.gauge('episode_length', 'Episode length')
        self.action_distribution = self.metrics_registry.counter('action_distribution', 'Action distribution', ['action'])
        
        # YOLO metrics
        self.yolo_fps = self.metrics_registry.gauge('yolo_fps', 'YOLO frames per second')
        self.yolo_inference_time = self.metrics_registry.histogram(
            'yolo_inference_time_seconds', 'YOLO inference time in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1]
        )
        self.detected_objects = self.metrics_registry.counter('detected_objects', 'Number of detected objects', ['class'])
        
        # Traffic metrics
        self.queue_length = self.metrics_registry.gauge('queue_length', 'Queue length', ['lane'])
        self.waiting_time = self.metrics_registry.gauge('waiting_time', 'Waiting time', ['lane'])
        self.throughput = self.metrics_registry.counter('throughput', 'Throughput', ['lane'])
    
    @contextmanager
    def measure_request_latency(self, endpoint: str) -> None:
        """Measure request latency.
        
        Args:
            endpoint: Endpoint name
            
        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            self.request_latency.labels(endpoint=endpoint).observe(latency)
            self.request_count.labels(endpoint=endpoint).inc()
    
    def record_quota(self, key: str, remaining: int) -> None:
        """Record remaining quota.
        
        Args:
            key: Quota key
            remaining: Remaining quota
        """
        self.quota_remaining.labels(key=key).set(remaining)
    
    def record_episode_metrics(self, reward: float, length: int) -> None:
        """Record episode metrics.
        
        Args:
            reward: Episode reward
            length: Episode length
        """
        self.episode_reward.set(reward)
        self.episode_length.set(length)
    
    def record_action(self, action: str) -> None:
        """Record an action.
        
        Args:
            action: Action name
        """
        self.action_distribution.labels(action=action).inc()
    
    def record_yolo_metrics(self, fps: float, inference_time: float) -> None:
        """Record YOLO metrics.
        
        Args:
            fps: Frames per second
            inference_time: Inference time in seconds
        """
        self.yolo_fps.set(fps)
        self.yolo_inference_time.observe(inference_time)
    
    def record_detection(self, object_class: str) -> None:
        """Record a detected object.
        
        Args:
            object_class: Object class
        """
        self.detected_objects.labels(class_=object_class).inc()
    
    def record_traffic_metrics(self, lane: str, queue_length: int, waiting_time: float, vehicles_passed: int) -> None:
        """Record traffic metrics.
        
        Args:
            lane: Lane ID
            queue_length: Queue length
            waiting_time: Waiting time
            vehicles_passed: Number of vehicles that passed
        """
        self.queue_length.labels(lane=lane).set(queue_length)
        self.waiting_time.labels(lane=lane).set(waiting_time)
        self.throughput.labels(lane=lane).inc(vehicles_passed)


def create_grafana_dashboard(dashboard_name: str, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a Grafana dashboard configuration.
    
    Args:
        dashboard_name: Dashboard name
        metrics: List of metric configurations
        
    Returns:
        Grafana dashboard configuration
    """
    dashboard = {
        "dashboard": {
            "id": None,
            "title": dashboard_name,
            "tags": ["adaptive-traffic"],
            "timezone": "browser",
            "editable": True,
            "panels": []
        },
        "overwrite": True
    }
    
    # Add panels for each metric
    y_pos = 0
    for i, metric in enumerate(metrics):
        panel = {
            "id": i + 1,
            "title": metric.get('title', f"Panel {i+1}"),
            "type": metric.get('type', 'graph'),
            "datasource": metric.get('datasource', 'Prometheus'),
            "gridPos": {
                "h": metric.get('height', 8),
                "w": metric.get('width', 12),
                "x": i % 2 * 12,
                "y": y_pos
            },
            "targets": [{
                "expr": metric.get('query', ''),
                "refId": "A"
            }]
        }
        
        dashboard["dashboard"]["panels"].append(panel)
        
        # Update y position for next row
        if i % 2 == 1:
            y_pos += 8
    
    return dashboard


def create_default_dashboards() -> Dict[str, Dict[str, Any]]:
    """Create default Grafana dashboards.
    
    Returns:
        Dictionary of dashboard configurations
    """
    dashboards = {}
    
    # System dashboard
    system_metrics = [
        {
            "title": "CPU Usage",
            "type": "gauge",
            "query": "adaptive_traffic_cpu_usage_percent",
            "height": 8,
            "width": 8
        },
        {
            "title": "Memory Usage",
            "type": "gauge",
            "query": "adaptive_traffic_memory_usage_percent",
            "height": 8,
            "width": 8
        },
        {
            "title": "Disk Usage",
            "type": "gauge",
            "query": "adaptive_traffic_disk_usage_percent",
            "height": 8,
            "width": 8
        },
        {
            "title": "GPU Usage",
            "type": "gauge",
            "query": "adaptive_traffic_gpu_usage_percent",
            "height": 8,
            "width": 8
        }
    ]
    dashboards['system'] = create_grafana_dashboard("System Metrics", system_metrics)
    
    # Application dashboard
    app_metrics = [
        {
            "title": "Request Count",
            "type": "graph",
            "query": "sum(rate(adaptive_traffic_request_count[5m])) by (endpoint)",
            "height": 8,
            "width": 12
        },
        {
            "title": "Request Latency",
            "type": "graph",
            "query": "histogram_quantile(0.95, sum(rate(adaptive_traffic_request_latency_seconds_bucket[5m])) by (endpoint, le))",
            "height": 8,
            "width": 12
        },
        {
            "title": "Quota Remaining",
            "type": "gauge",
            "query": "adaptive_traffic_quota_remaining",
            "height": 8,
            "width": 8
        },
        {
            "title": "YOLO FPS",
            "type": "gauge",
            "query": "adaptive_traffic_yolo_fps",
            "height": 8,
            "width": 8
        }
    ]
    dashboards['application'] = create_grafana_dashboard("Application Metrics", app_metrics)
    
    # RL dashboard
    rl_metrics = [
        {
            "title": "Episode Reward",
            "type": "graph",
            "query": "adaptive_traffic_episode_reward",
            "height": 8,
            "width": 12
        },
        {
            "title": "Episode Length",
            "type": "graph",
            "query": "adaptive_traffic_episode_length",
            "height": 8,
            "width": 12
        },
        {
            "title": "Action Distribution",
            "type": "pie",
            "query": "sum(adaptive_traffic_action_distribution) by (action)",
            "height": 8,
            "width": 12
        }
    ]
    dashboards['rl'] = create_grafana_dashboard("RL Metrics", rl_metrics)
    
    # Traffic dashboard
    traffic_metrics = [
        {
            "title": "Queue Length",
            "type": "graph",
            "query": "adaptive_traffic_queue_length",
            "height": 8,
            "width": 12
        },
        {
            "title": "Waiting Time",
            "type": "graph",
            "query": "adaptive_traffic_waiting_time",
            "height": 8,
            "width": 12
        },
        {
            "title": "Throughput",
            "type": "graph",
            "query": "sum(rate(adaptive_traffic_throughput[5m])) by (lane)",
            "height": 8,
            "width": 12
        }
    ]
    dashboards['traffic'] = create_grafana_dashboard("Traffic Metrics", traffic_metrics)
    
    return dashboards