"""
Load Testing Script for API Endpoints using Locust.

Tests API performance under various load conditions.
"""

from locust import HttpUser, task, between
from typing import Dict
import random


class TrafficControlAPIUser(HttpUser):
    """
    Locust user class for simulating API traffic.
    
    Simulates realistic user behavior:
    - Making traffic decisions
    - Checking system health
    - Fetching metrics
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts."""
        # Health check on startup
        self.client.get("/health")
    
    @task(5)
    def make_traffic_decision(self):
        """Make a traffic decision (high frequency)."""
        payload = {
            "intersection_id": f"int-{random.randint(1, 100):03d}",
            "queue_lengths": [
                random.uniform(5, 20) for _ in range(4)
            ],
            "wait_times": [
                random.uniform(10, 40) for _ in range(4)
            ],
            "throughput": random.uniform(300, 600),
            "current_phase": random.randint(0, 3),
        }
        
        self.client.post(
            "/api/v1/traffic/decision",
            json=payload,
            name="POST /api/v1/traffic/decision",
        )
    
    @task(2)
    def get_system_health(self):
        """Check system health (medium frequency)."""
        self.client.get(
            "/api/v1/system/health",
            name="GET /api/v1/system/health",
        )
    
    @task(2)
    def get_dashboard_metrics(self):
        """Get dashboard metrics (medium frequency)."""
        self.client.get(
            "/api/v1/metrics/dashboard",
            name="GET /api/v1/metrics/dashboard",
        )
    
    @task(1)
    def get_kpi_metrics(self):
        """Get KPI metrics (low frequency)."""
        self.client.get(
            "/api/v1/metrics/kpis",
            name="GET /api/v1/metrics/kpis",
        )
    
    @task(1)
    def get_algorithm_performance(self):
        """Get algorithm performance (low frequency)."""
        self.client.get(
            "/api/v1/analytics/algorithm-performance",
            name="GET /api/v1/analytics/algorithm-performance",
        )
    
    @task(1)
    def get_intersections(self):
        """Get all intersections (low frequency)."""
        self.client.get(
            "/api/v1/traffic/intersections",
            name="GET /api/v1/traffic/intersections",
        )


class BatchTrafficUser(HttpUser):
    """
    User class for batch traffic decision requests.
    
    Simulates batch processing scenarios.
    """
    
    wait_time = between(5, 10)  # Longer wait time for batch operations
    
    @task(3)
    def batch_decision(self):
        """Make batch traffic decisions."""
        payload = {
            "requests": [
                {
                    "intersection_id": f"int-{random.randint(1, 100):03d}",
                    "queue_lengths": [
                        random.uniform(5, 20) for _ in range(4)
                    ],
                    "wait_times": [
                        random.uniform(10, 40) for _ in range(4)
                    ],
                    "throughput": random.uniform(300, 600),
                    "current_phase": random.randint(0, 3),
                }
                for _ in range(random.randint(2, 5))
            ]
        }
        
        self.client.post(
            "/api/v1/traffic/batch",
            json=payload,
            name="POST /api/v1/traffic/batch",
        )

