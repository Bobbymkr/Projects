# Tutorials
## Adaptive Traffic Signal Control System

Step-by-step tutorials for common development tasks.

---

## Table of Contents

1. [Quick Start Tutorial](#quick-start-tutorial)
2. [Algorithm Implementation Guide](#algorithm-implementation-guide)
3. [Dashboard Integration Tutorial](#dashboard-integration-tutorial)
4. [Testing Tutorial](#testing-tutorial)

---

## Quick Start Tutorial

### Getting Started in 5 Minutes

This tutorial will guide you through:
1. Setting up your development environment
2. Running your first traffic simulation
3. Making your first API call
4. Viewing results in the dashboard

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd adaptive_traffic

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start the API Server

```bash
# Start API server
python scripts/start_api.py

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/api/docs
```

### Step 3: Make Your First API Call

```python
import requests

# Get access token
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "admin", "password": "secret"}
)
token = response.json()["access_token"]

# Make a traffic decision
decision = requests.post(
    "http://localhost:8000/api/v1/traffic/decision",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "intersection_id": "intersection_1",
        "current_state": {
            "queue_lengths": [5, 3, 8, 2],
            "wait_times": [12.5, 8.3, 15.2, 6.1],
            "arrival_rates": [0.3, 0.2, 0.4, 0.15]
        },
        "algorithm": "dqn"
    }
)

print(decision.json())
```

### Step 4: View Dashboard

```bash
# Start dashboard
cd dashboard
npm install
npm start

# Dashboard will be available at http://localhost:3000
```

---

## Algorithm Implementation Guide

### Implementing a New Control Algorithm

This guide shows you how to add a new traffic control algorithm.

### Step 1: Create Algorithm Class

Create a new file `src/control/my_algorithm.py`:

```python
from typing import List, Dict, Any
from src.control.base import BaseController

class MyAlgorithm(BaseController):
    """
    Custom traffic control algorithm.
    
    Implements BaseController interface for integration
    with the traffic control system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize algorithm with configuration."""
        super().__init__(config)
        self.name = "my_algorithm"
        
    def select_phase(
        self,
        queue_lengths: List[int],
        wait_times: List[float],
        arrival_rates: List[float]
    ) -> int:
        """
        Select optimal phase based on current state.
        
        Args:
            queue_lengths: Current queue lengths for each lane
            wait_times: Current wait times for each lane
            arrival_rates: Vehicle arrival rates
            
        Returns:
            Selected phase (0-3)
        """
        # Your algorithm logic here
        # Example: Select phase with highest queue
        max_queue_idx = queue_lengths.index(max(queue_lengths))
        return max_queue_idx
    
    def calculate_duration(
        self,
        phase: int,
        queue_length: int,
        arrival_rate: float
    ) -> float:
        """
        Calculate phase duration.
        
        Args:
            phase: Selected phase
            queue_length: Queue length for phase
            arrival_rate: Arrival rate for phase
            
        Returns:
            Phase duration in seconds
        """
        # Calculate duration based on queue
        base_duration = 30.0
        queue_factor = queue_length * 2.0
        return min(base_duration + queue_factor, 120.0)
```

### Step 2: Register Algorithm

Add to `src/control/__init__.py`:

```python
from .my_algorithm import MyAlgorithm

__all__ = [
    "MyAlgorithm",
    # ... other algorithms
]
```

### Step 3: Integrate with Service

Update `src/api/services/traffic_service.py`:

```python
from src.control import MyAlgorithm

class TrafficService:
    def __init__(self):
        self.algorithms = {
            "my_algorithm": MyAlgorithm,
            # ... other algorithms
        }
```

### Step 4: Test Your Algorithm

Create test file `tests/control/test_my_algorithm.py`:

```python
import pytest
from src.control import MyAlgorithm

def test_my_algorithm_phase_selection():
    """Test phase selection."""
    algorithm = MyAlgorithm()
    phase = algorithm.select_phase(
        queue_lengths=[5, 3, 8, 2],
        wait_times=[12.5, 8.3, 15.2, 6.1],
        arrival_rates=[0.3, 0.2, 0.4, 0.15]
    )
    assert 0 <= phase < 4

def test_my_algorithm_duration_calculation():
    """Test duration calculation."""
    algorithm = MyAlgorithm()
    duration = algorithm.calculate_duration(
        phase=0,
        queue_length=5,
        arrival_rate=0.3
    )
    assert 15.0 <= duration <= 120.0
```

### Step 5: Use in API

Your algorithm is now available via API:

```bash
curl -X POST "http://localhost:8000/api/v1/traffic/decision" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "intersection_id": "intersection_1",
    "current_state": { ... },
    "algorithm": "my_algorithm"
  }'
```

---

## Dashboard Integration Tutorial

### Adding a New Dashboard Component

This tutorial shows how to add a new component to the React dashboard.

### Step 1: Create Component

Create `dashboard/src/components/MyComponent.tsx`:

```typescript
import React, { useEffect, useState } from 'react';
import { Card, Spin } from 'antd';
import { api } from '../services/api';

interface MyComponentProps {
  intersectionId: string;
}

export const MyComponent: React.FC<MyComponentProps> = ({ intersectionId }) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await api.get(`/traffic/intersections/${intersectionId}`);
        setData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Poll every 5 seconds
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [intersectionId]);

  if (loading) {
    return <Spin />;
  }

  return (
    <Card title="My Component">
      {/* Your component content */}
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </Card>
  );
};
```

### Step 2: Add to Dashboard

Update `dashboard/src/pages/Dashboard.tsx`:

```typescript
import { MyComponent } from '../components/MyComponent';

export const Dashboard: React.FC = () => {
  return (
    <div>
      {/* Other components */}
      <MyComponent intersectionId="intersection_1" />
    </div>
  );
};
```

### Step 3: Real-Time Updates (Optional)

For WebSocket updates:

```typescript
import { useEffect, useState } from 'react';
import { wsClient } from '../services/websocket';

export const MyRealtimeComponent: React.FC = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const ws = wsClient.connect('/ws/traffic');
    
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setData(update);
    };

    return () => {
      ws.close();
    };
  }, []);

  return <div>{/* Render data */}</div>;
};
```

---

## Testing Tutorial

### Writing Comprehensive Tests

This tutorial covers writing tests for the traffic control system.

### Step 1: Unit Tests

Test individual functions and classes:

```python
# tests/unit/control/test_fuzzy_control.py
import pytest
from src.control.fuzzy_control import FuzzyController

def test_fuzzy_controller_initialization():
    """Test controller initialization."""
    controller = FuzzyController()
    assert controller is not None

def test_fuzzy_controller_phase_selection():
    """Test phase selection logic."""
    controller = FuzzyController()
    phase = controller.select_phase(
        queue_lengths=[5, 3, 8, 2],
        wait_times=[12.5, 8.3, 15.2, 6.1]
    )
    assert 0 <= phase < 4
```

### Step 2: Integration Tests

Test component interactions:

```python
# tests/integration/test_traffic_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_traffic_decision_endpoint():
    """Test traffic decision API endpoint."""
    response = client.post(
        "/api/v1/traffic/decision",
        json={
            "intersection_id": "intersection_1",
            "current_state": {
                "queue_lengths": [5, 3, 8, 2],
                "wait_times": [12.5, 8.3, 15.2, 6.1]
            },
            "algorithm": "fuzzy"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
```

### Step 3: Performance Tests

Test performance and load:

```python
# tests/performance/test_decision_latency.py
import time
import pytest
from src.api.services import TrafficService

def test_decision_latency():
    """Test decision making latency."""
    service = TrafficService()
    state = {
        "queue_lengths": [5, 3, 8, 2],
        "wait_times": [12.5, 8.3, 15.2, 6.1]
    }
    
    start = time.time()
    decision = service.make_decision("intersection_1", state, "dqn")
    duration = time.time() - start
    
    assert duration < 0.1  # Should be < 100ms
    assert decision is not None
```

### Step 4: End-to-End Tests

Test complete workflows:

```python
# tests/e2e/test_traffic_control_flow.py
import pytest

def test_complete_traffic_control_flow():
    """Test complete traffic control workflow."""
    # 1. Initialize system
    # 2. Make decision
    # 3. Verify decision
    # 4. Check metrics
    # 5. Verify dashboard updates
    pass
```

### Step 5: Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/unit/control/test_fuzzy_control.py

# Run with verbose output
pytest -v tests/
```

---

## Additional Resources

- [Getting Started Guide](GETTING_STARTED.md)
- [API Documentation](../api/API_DOCUMENTATION.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Best Practices](BEST_PRACTICES.md)

---

**Last Updated**: November 30, 2024

