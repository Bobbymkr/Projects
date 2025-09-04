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
    
    print("Observability components set up successfully!")


if __name__ == "__main__":
    main()