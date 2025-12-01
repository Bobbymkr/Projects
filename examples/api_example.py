"""
API Usage Examples.

Complete examples for using the Adaptive Traffic Control API.
"""

import requests
import time
from typing import Dict, Any, Optional


class TrafficAPIClient:
    """Client for interacting with Traffic Control API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        """Initialize API client."""
        self.base_url = base_url
        self.token: Optional[str] = None
    
    def authenticate(self, username: str = "admin", password: str = "secret") -> bool:
        """
        Authenticate and get access token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authentication successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                data={"username": username, "password": password}
            )
            response.raise_for_status()
            self.token = response.json()["access_token"]
            return True
        except requests.RequestException as e:
            print(f"Authentication failed: {e}")
            return False
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}
    
    def make_decision(
        self,
        intersection_id: str,
        current_state: Dict[str, Any],
        algorithm: str = "dqn"
    ) -> Optional[Dict[str, Any]]:
        """
        Make a traffic signal decision.
        
        Args:
            intersection_id: Intersection identifier
            current_state: Current traffic state
            algorithm: Algorithm to use
            
        Returns:
            Decision response or None if error
        """
        try:
            response = requests.post(
                f"{self.base_url}/traffic/decision",
                headers=self.headers,
                json={
                    "intersection_id": intersection_id,
                    "current_state": current_state,
                    "algorithm": algorithm,
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Decision request failed: {e}")
            return None
    
    def get_intersections(self) -> Optional[Dict[str, Any]]:
        """Get list of all intersections."""
        try:
            response = requests.get(
                f"{self.base_url}/traffic/intersections",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to get intersections: {e}")
            return None
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics."""
        try:
            response = requests.get(
                f"{self.base_url}/metrics/kpis",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to get metrics: {e}")
            return None
    
    def get_health(self) -> Optional[Dict[str, Any]]:
        """Get system health status."""
        try:
            response = requests.get(f"{self.base_url}/../health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Health check failed: {e}")
            return None


def example_basic_usage():
    """Basic API usage example."""
    print("=== Basic API Usage Example ===\n")
    
    # Initialize client
    client = TrafficAPIClient()
    
    # Authenticate
    if not client.authenticate():
        print("Failed to authenticate. Please check credentials.")
        return
    
    print("✓ Authenticated successfully\n")
    
    # Make a decision
    current_state = {
        "queue_lengths": [5, 3, 8, 2],
        "wait_times": [12.5, 8.3, 15.2, 6.1],
        "arrival_rates": [0.3, 0.2, 0.4, 0.15],
    }
    
    decision = client.make_decision("intersection_1", current_state, algorithm="dqn")
    if decision:
        print("Decision made:")
        print(f"  Phase: {decision['decision']['phase']}")
        print(f"  Duration: {decision['decision']['duration']}s")
        print(f"  Confidence: {decision['decision']['confidence']:.2%}")
        print(f"  Expected wait reduction: {decision['performance']['expected_wait_reduction']:.2%}\n")


def example_continuous_monitoring():
    """Example of continuous monitoring."""
    print("=== Continuous Monitoring Example ===\n")
    
    client = TrafficAPIClient()
    if not client.authenticate():
        return
    
    # Monitor for 10 iterations
    for i in range(10):
        print(f"Iteration {i+1}:")
        
        # Get metrics
        metrics = client.get_metrics()
        if metrics:
            kpis = metrics.get("kpis", {})
            print(f"  Average wait time: {kpis.get('average_wait_time', 'N/A')}s")
            print(f"  Efficiency score: {kpis.get('efficiency_score', 'N/A')}")
        
        time.sleep(2)
    
    print("\nMonitoring complete.")


def example_batch_processing():
    """Example of batch processing multiple intersections."""
    print("=== Batch Processing Example ===\n")
    
    client = TrafficAPIClient()
    if not client.authenticate():
        return
    
    # Get all intersections
    intersections = client.get_intersections()
    if not intersections:
        return
    
    intersection_list = intersections.get("intersections", [])
    print(f"Processing {len(intersection_list)} intersections...\n")
    
    # Process each intersection
    for intersection in intersection_list[:5]:  # Limit to first 5
        intersection_id = intersection["id"]
        
        # Make decision for each
        current_state = {
            "queue_lengths": [5, 3, 8, 2],
            "wait_times": [12.5, 8.3, 15.2, 6.1],
            "arrival_rates": [0.3, 0.2, 0.4, 0.15],
        }
        
        decision = client.make_decision(intersection_id, current_state)
        if decision:
            print(f"{intersection_id}: Phase {decision['decision']['phase']}, "
                  f"Duration {decision['decision']['duration']}s")


def example_health_monitoring():
    """Example of health monitoring."""
    print("=== Health Monitoring Example ===\n")
    
    client = TrafficAPIClient()
    
    # Check health
    health = client.get_health()
    if health:
        print("System Health:")
        print(f"  Status: {health.get('status')}")
        print(f"  Version: {health.get('version')}")
        print(f"  Uptime: {health.get('uptime_seconds', 0) / 3600:.2f} hours")
        
        components = health.get("components", {})
        print("\nComponents:")
        for component, status in components.items():
            print(f"  {component}: {status}")


if __name__ == "__main__":
    print("Traffic Control API Examples\n")
    print("="*50 + "\n")
    
    # Run examples
    example_basic_usage()
    print("\n" + "="*50 + "\n")
    
    example_health_monitoring()
    print("\n" + "="*50 + "\n")
    
    example_batch_processing()
    
    # Uncomment to run continuous monitoring
    # example_continuous_monitoring()

