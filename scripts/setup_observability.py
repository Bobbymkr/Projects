#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup observability components for the Adaptive Traffic project.

This script creates utilities for metrics, tracing, and dashboards.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def create_metrics_utility():
    """
    Create a metrics.py utility in src/utils for Prometheus, W&B, MLflow, and TensorBoard.
    """
    # Ensure the utils directory exists
    utils_dir = project_root / "src" / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the metrics.py file
    metrics_path = utils_dir / "metrics.py"
    
    metrics_content = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics utilities for the Adaptive Traffic project.

This module provides utilities for collecting and reporting metrics using
Prometheus, Weights & Biases, MLflow, and TensorBoard.
"""

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Import optional dependencies
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import tensorboardX
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Load configuration
config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "observability.json"
if config_path.exists():
    with open(config_path, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        "prometheus": {"enabled": False, "port": 8000},
        "wandb": {"enabled": False, "project": "adaptive_traffic"},
        "mlflow": {"enabled": False, "tracking_uri": "http://localhost:5000"},
        "tensorboard": {"enabled": False, "log_dir": "logs/tensorboard"}
    }

# Global registry for metrics
_METRICS = {}
_WRITERS = {}


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str):
        """Initialize the timer.
        
        Args:
            name: The name of the timer for reporting.
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and report the duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Report to Prometheus if available
        if PROMETHEUS_AVAILABLE and CONFIG["prometheus"]["enabled"]:
            histogram = get_metric(self.name, "histogram")
            histogram.observe(duration)
        
        # Report to experiment trackers
        log_metric(self.name, duration)
        
        return False  # Don't suppress exceptions


def time_function(name: Optional[str] = None):
    """Decorator for timing function execution.
    
    Args:
        name: Optional name for the timer. If not provided, uses the function name.
    
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            with Timer(timer_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_metric(name: str, metric_type: str, description: str = "", labels: List[str] = None) -> Any:
    """Get or create a Prometheus metric.
    
    Args:
        name: The name of the metric.
        metric_type: The type of metric (counter, gauge, histogram, summary).
        description: Optional description of the metric.
        labels: Optional list of label names for the metric.
    
    Returns:
        The Prometheus metric object.
    """
    if not PROMETHEUS_AVAILABLE or not CONFIG["prometheus"]["enabled"]:
        return None
    
    labels = labels or []
    key = f"{name}_{metric_type}"
    
    if key in _METRICS:
        return _METRICS[key]
    
    if metric_type == "counter":
        metric = prom.Counter(name, description, labels)
    elif metric_type == "gauge":
        metric = prom.Gauge(name, description, labels)
    elif metric_type == "histogram":
        metric = prom.Histogram(name, description, labels)
    elif metric_type == "summary":
        metric = prom.Summary(name, description, labels)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    _METRICS[key] = metric
    return metric


def create_counter(name: str, description: str = "", labels: List[str] = None) -> Any:
    """Create a counter metric.
    
    Args:
        name: The name of the counter.
        description: Optional description of the counter.
        labels: Optional list of label names for the counter.
    
    Returns:
        The counter metric object.
    """
    return get_metric(name, "counter", description, labels)


def create_gauge(name: str, description: str = "", labels: List[str] = None) -> Any:
    """Create a gauge metric.
    
    Args:
        name: The name of the gauge.
        description: Optional description of the gauge.
        labels: Optional list of label names for the gauge.
    
    Returns:
        The gauge metric object.
    """
    return get_metric(name, "gauge", description, labels)


def create_histogram(name: str, description: str = "", labels: List[str] = None) -> Any:
    """Create a histogram metric.
    
    Args:
        name: The name of the histogram.
        description: Optional description of the histogram.
        labels: Optional list of label names for the histogram.
    
    Returns:
        The histogram metric object.
    """
    return get_metric(name, "histogram", description, labels)


def create_summary(name: str, description: str = "", labels: List[str] = None) -> Any:
    """Create a summary metric.
    
    Args:
        name: The name of the summary.
        description: Optional description of the summary.
        labels: Optional list of label names for the summary.
    
    Returns:
        The summary metric object.
    """
    return get_metric(name, "summary", description, labels)


def increment_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
    """Increment a counter metric.
    
    Args:
        name: The name of the counter.
        value: The value to increment by.
        labels: Optional dictionary of label values.
    """
    if not PROMETHEUS_AVAILABLE or not CONFIG["prometheus"]["enabled"]:
        return
    
    counter = get_metric(name, "counter")
    if labels:
        counter.labels(**labels).inc(value)
    else:
        counter.inc(value)
    
    # Also log to experiment trackers
    log_metric(f"{name}_total", value, labels)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None) -> None:
    """Set a gauge metric.
    
    Args:
        name: The name of the gauge.
        value: The value to set.
        labels: Optional dictionary of label values.
    """
    if not PROMETHEUS_AVAILABLE or not CONFIG["prometheus"]["enabled"]:
        return
    
    gauge = get_metric(name, "gauge")
    if labels:
        gauge.labels(**labels).set(value)
    else:
        gauge.set(value)
    
    # Also log to experiment trackers
    log_metric(name, value, labels)


def observe_histogram(name: str, value: float, labels: Dict[str, str] = None) -> None:
    """Observe a value for a histogram metric.
    
    Args:
        name: The name of the histogram.
        value: The value to observe.
        labels: Optional dictionary of label values.
    """
    if not PROMETHEUS_AVAILABLE or not CONFIG["prometheus"]["enabled"]:
        return
    
    histogram = get_metric(name, "histogram")
    if labels:
        histogram.labels(**labels).observe(value)
    else:
        histogram.observe(value)
    
    # Also log to experiment trackers
    log_metric(name, value, labels)
"""
    
    with open(metrics_path, "w") as f:
        f.write(metrics_content)
    
    print(f"Created metrics utility at {metrics_path}")
    return metrics_path


def create_tracing_utility():
    """
    Create a tracing.py utility in src/utils for OpenTelemetry.
    """
    # Ensure the utils directory exists
    utils_dir = project_root / "src" / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the tracing.py file
    tracing_path = utils_dir / "tracing.py"
    
    tracing_content = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tracing utilities for the Adaptive Traffic project.

This module provides utilities for distributed tracing using OpenTelemetry.
"""

import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Import optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Load configuration
config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "observability.json"
if config_path.exists():
    with open(config_path, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        "opentelemetry": {
            "enabled": False,
            "service_name": "adaptive_traffic",
            "exporter": "console",
            "endpoint": None
        }
    }

_TRACER = None


def init_tracer() -> Optional[trace.Tracer]:
    """Initialize the OpenTelemetry tracer.
    
    Returns:
        The tracer object, or None if not enabled.
    """
    global _TRACER
    if _TRACER:
        return _TRACER
    
    if not OPENTELEMETRY_AVAILABLE or not CONFIG["opentelemetry"]["enabled"]:
        return None
    
    provider = TracerProvider()
    
    exporter_type = CONFIG["opentelemetry"].get("exporter", "console")
    if exporter_type == "console":
        exporter = ConsoleSpanExporter()
    else:
        # Add other exporters like Jaeger, Zipkin here
        exporter = ConsoleSpanExporter()
    
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    service_name = CONFIG["opentelemetry"].get("service_name", "adaptive_traffic")
    _TRACER = trace.get_tracer(service_name)
    
    return _TRACER


def get_tracer() -> Optional[trace.Tracer]:
    """Get the initialized tracer.
    
    Returns:
        The tracer object, or None if not initialized.
    """
    return _TRACER or init_tracer()


def trace_function(name: Optional[str] = None):
    """Decorator for tracing function execution.
    
    Args:
        name: Optional name for the span. If not provided, uses the function name.
    
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer:
                return func(*args, **kwargs)
            
            span_name = name or f"{func.__module__}.{func.__name__}"
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
"""
    
    with open(tracing_path, "w") as f:
        f.write(tracing_content)
    
    print(f"Created tracing utility at {tracing_path}")
    return tracing_path


def create_dashboard_utility():
    """
    Create a dashboard.py utility in src/utils for Grafana.
    """
    # Ensure the utils directory exists
    utils_dir = project_root / "src" / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the dashboard.py file
    dashboard_path = utils_dir / "dashboard.py"
    
    dashboard_content = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard utilities for the Adaptive Traffic project.

This module provides utilities for creating and managing Grafana dashboards.
"""

import json
from pathlib import Path

# Import optional dependencies
try:
    from grafana_api.grafana_face import GrafanaFace
    GRAFANA_AVAILABLE = True
except ImportError:
    GRAFANA_AVAILABLE = False

# Load configuration
config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "observability.json"
if config_path.exists():
    with open(config_path, "r") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        "grafana": {
            "enabled": False,
            "host": "localhost",
            "port": 3000,
            "username": "admin",
            "password": "admin"
        }
    }

_GRAFANA_CLIENT = None


def init_grafana() -> GrafanaFace:
    """Initialize the Grafana API client."""
    global _GRAFANA_CLIENT
    if _GRAFANA_CLIENT:
        return _GRAFANA_CLIENT
    
    if not GRAFANA_AVAILABLE or not CONFIG["grafana"]["enabled"]:
        return None
    
    try:
        _GRAFANA_CLIENT = GrafanaFace(
            auth=(CONFIG["grafana"]["username"], CONFIG["grafana"]["password"]),
            host=CONFIG["grafana"]["host"],
            port=CONFIG["grafana"]["port"]
        )
    except Exception as e:
        print(f"Failed to connect to Grafana: {e}")
        return None
    
    return _GRAFANA_CLIENT


def create_traffic_dashboard():
    """Create a default traffic dashboard in Grafana."""
    client = init_grafana()
    if not client:
        print("Grafana is not enabled or available.")
        return
    
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "Adaptive Traffic Dashboard",
            "tags": ["traffic", "sumo"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "25s"
        },
        "folderId": 0,
        "overwrite": False
    }
    
    try:
        client.dashboard.update_dashboard(dashboard=dashboard)
        print("Successfully created/updated Grafana dashboard.")
    except Exception as e:
        print(f"Failed to create/update Grafana dashboard: {e}")

"""
    
    with open(dashboard_path, "w") as f:
        f.write(dashboard_content)
    
    print(f"Created dashboard utility at {dashboard_path}")
    return dashboard_path


def create_observability_example():
    """
    Create an observability_example.py in the examples directory.
    """
    # Ensure the examples directory exists
    examples_dir = project_root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the example file
    example_path = examples_dir / "observability_example.py"
    
    example_content = """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of observability components.
"""

import time
import random

from src.utils.metrics import (
    Timer,
    time_function,
    increment_counter,
    set_gauge,
    observe_histogram,
)
from src.utils.tracing import trace_function, get_tracer


@trace_function()
@time_function()
def process_data():
    """Simulate some data processing."""
    print("Processing data...")
    # Simulate work
    time.sleep(random.uniform(0.1, 0.5))
    
    # Increment a counter
    increment_counter("data_processed_total", labels={"type": "example"})
    
    # Set a gauge
    set_gauge("active_workers", random.randint(1, 10))
    
    # Observe a histogram
    observe_histogram("processing_latency_seconds", random.random())
    
    print("Data processing complete.")


def main():
    """Main function to run the example."""
    print("Running observability example...")
    
    # Initialize tracer
    tracer = get_tracer()
    
    if tracer:
        with tracer.start_as_current_span("main_span"):
            for i in range(5):
                process_data()
    else:
        for i in range(5):
            process_data()
            
    print("Observability example complete.")


if __name__ == "__main__":
    main()
"""
    
    with open(example_path, "w") as f:
        f.write(example_content)
    
    print(f"Created observability example at {example_path}")
    return example_path


def create_observability_config():
    """
    Create an observability.json configuration file in the configs directory.
    """
    # Ensure the configs directory exists
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the observability.json file
    config_path = configs_dir / "observability.json"
    
    config = {
        "prometheus": {
            "enabled": True,
            "port": 8000,
            "path": "/metrics"
        },
        "wandb": {
            "enabled": False,
            "project": "adaptive_traffic",
            "entity": None,
            "tags": ["traffic", "reinforcement-learning", "computer-vision"]
        },
        "mlflow": {
            "enabled": False,
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "adaptive_traffic"
        },
        "tensorboard": {
            "enabled": True,
            "log_dir": "logs/tensorboard"
        },
        "opentelemetry": {
            "enabled": True,
            "service_name": "adaptive_traffic",
            "exporter": "console",  # Options: console, jaeger, zipkin
            "endpoint": None
        },
        "grafana": {
            "enabled": False,
            "host": "localhost",
            "port": 3000,
            "username": "admin",
            "password": "admin"
        }
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created observability config at {config_path}")
    return config_path


def main():
    """
    Main function to set up observability components.
    """
    print("Setting up observability components...")
    
    # Create the observability config
    create_observability_config()
    
    # Create the metrics utility
    create_metrics_utility()
    
    # Create the tracing utility
    create_tracing_utility()
    
    # Create the dashboard utility
    create_dashboard_utility()
    
    # Create the observability example
    create_observability_example()
    
    print("Observability components set up successfully!")


if __name__ == "__main__":
    main()