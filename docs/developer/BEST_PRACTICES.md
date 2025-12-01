# Best Practices Guide
## Adaptive Traffic Signal Control System

Development best practices and coding standards for maintaining high-quality code.

---

## Code Style

### Python Style Guide

We follow **PEP 8** with the following modifications:

- **Line Length**: Maximum 100 characters
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all public functions/classes

#### Example

```python
from typing import List, Optional, Dict, Any

def calculate_wait_time(
    queue_lengths: List[int],
    arrival_rate: float,
    algorithm: str = "dqn"
) -> float:
    """
    Calculate expected wait time for traffic signal.
    
    Args:
        queue_lengths: Current queue lengths for each lane
        arrival_rate: Vehicles per second arrival rate
        algorithm: Algorithm to use for calculation
        
    Returns:
        Expected wait time in seconds
        
    Raises:
        ValueError: If queue_lengths is empty
    """
    if not queue_lengths:
        raise ValueError("queue_lengths cannot be empty")
    
    # Implementation
    return wait_time
```

---

### Type Safety

Always use type hints:

```python
# Good
def process_intersection(
    intersection_id: str,
    state: Dict[str, Any],
    config: Optional[Config] = None
) -> Decision:
    pass

# Bad
def process_intersection(intersection_id, state, config=None):
    pass
```

---

## Code Organization

### Module Structure

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── traffic.py
│   │   └── ...
│   └── services/
├── rl/
│   ├── __init__.py
│   ├── agents/
│   └── environments/
└── ...
```

### Import Organization

```python
# Standard library imports
import logging
from typing import List, Optional
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from fastapi import APIRouter, Depends

# Local application imports
from src.api.services import TrafficService
from src.rl.agents import DQNAgent
```

---

## Error Handling

### Exception Handling Best Practices

1. **Be Specific**: Use specific exception types

```python
# Good
try:
    decision = agent.select_action(state)
except ModelNotLoadedError as e:
    logger.error(f"Model not loaded: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid state: {e}")
    return default_decision()

# Bad
try:
    decision = agent.select_action(state)
except Exception as e:
    logger.error(f"Error: {e}")
    return None
```

2. **Log Appropriately**: Log at the right level

```python
# DEBUG: Detailed information for debugging
logger.debug(f"Processing state: {state}")

# INFO: General informational messages
logger.info(f"Decision made: {decision}")

# WARNING: Warning messages for unusual situations
logger.warning(f"High wait time detected: {wait_time}s")

# ERROR: Error messages for failures
logger.error(f"Failed to load model: {e}", exc_info=True)
```

3. **Use Custom Exceptions**: Create domain-specific exceptions

```python
class TrafficControlError(Exception):
    """Base exception for traffic control errors."""
    pass

class InvalidIntersectionError(TrafficControlError):
    """Raised when intersection ID is invalid."""
    pass

class ModelNotLoadedError(TrafficControlError):
    """Raised when ML model is not loaded."""
    pass
```

---

## Testing Best Practices

### Test Structure

Follow **Arrange-Act-Assert** pattern:

```python
def test_dqn_agent_action_selection():
    """Test DQN agent selects valid actions."""
    # Arrange
    agent = DQNAgent(state_dim=10, action_dim=4)
    state = np.random.rand(10)
    
    # Act
    action = agent.select_action(state, epsilon=0.0)
    
    # Assert
    assert 0 <= action < 4
    assert isinstance(action, (int, np.integer))
```

### Test Naming

- Use descriptive test names
- Prefix with `test_`
- Describe what is being tested

```python
# Good
def test_dqn_agent_selects_valid_action_for_normal_state():
    pass

def test_dqn_agent_handles_empty_state_gracefully():
    pass

# Bad
def test_agent():
    pass

def test_1():
    pass
```

### Test Coverage

- Aim for **90%+ code coverage**
- Focus on critical paths
- Test edge cases and error conditions
- Use fixtures for common test data

```python
import pytest

@pytest.fixture
def sample_traffic_state():
    """Fixture for sample traffic state."""
    return {
        "queue_lengths": [5, 3, 8, 2],
        "wait_times": [12.5, 8.3, 15.2, 6.1],
        "arrival_rates": [0.3, 0.2, 0.4, 0.15]
    }

def test_traffic_decision_with_fixture(sample_traffic_state):
    """Test traffic decision using fixture."""
    decision = make_decision(sample_traffic_state)
    assert decision is not None
```

---

## Documentation Standards

### Docstrings

Use Google-style docstrings:

```python
def train_dqn_agent(
    episodes: int,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> DQNAgent:
    """
    Train a DQN agent on traffic control environment.
    
    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        
    Returns:
        Trained DQN agent
        
    Raises:
        ValueError: If episodes < 1
        RuntimeError: If training fails
        
    Example:
        >>> agent = train_dqn_agent(episodes=100, learning_rate=0.001)
        >>> decision = agent.select_action(state)
    """
    pass
```

### Code Comments

- **Explain Why, Not What**: Comments should explain intent
- **Keep Comments Updated**: Update comments when code changes
- **Avoid Obvious Comments**: Code should be self-documenting

```python
# Good
# Use epsilon-greedy exploration to balance exploitation and exploration
action = agent.select_action(state, epsilon=0.1)

# Bad
# Select action from agent
action = agent.select_action(state, epsilon=0.1)
```

---

## Performance Optimization

### Async/Await

Use async/await for I/O-bound operations:

```python
# Good
async def fetch_intersection_data(intersection_id: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/intersections/{intersection_id}") as resp:
            return await resp.json()

# Bad
def fetch_intersection_data(intersection_id: str) -> Dict:
    response = requests.get(f"/api/intersections/{intersection_id}")
    return response.json()
```

### Caching

Cache expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_optimal_phase(intersection_id: str, state_hash: int) -> int:
    """Cache expensive phase calculations."""
    # Expensive computation
    return optimal_phase
```

### Vectorization

Use NumPy for numerical operations:

```python
# Good
import numpy as np

wait_times = np.array(queue_lengths) / arrival_rates

# Bad
wait_times = [q / arr_rate for q, arr_rate in zip(queue_lengths, arrival_rates)]
```

---

## Security Best Practices

### Input Validation

Always validate and sanitize inputs:

```python
from pydantic import BaseModel, validator

class TrafficDecisionRequest(BaseModel):
    intersection_id: str
    queue_lengths: List[int]
    
    @validator('queue_lengths')
    def validate_queue_lengths(cls, v):
        if not v or len(v) != 4:
            raise ValueError("Must provide exactly 4 queue lengths")
        if any(q < 0 for q in v):
            raise ValueError("Queue lengths must be non-negative")
        return v
```

### Secrets Management

Never commit secrets to version control:

```python
# Good - Use environment variables
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    secret_key: str = os.getenv("SECRET_KEY")
    database_url: str = os.getenv("DATABASE_URL")

# Bad - Hardcoded secrets
SECRET_KEY = "my-secret-key-12345"
```

### SQL Injection Prevention

Use parameterized queries or ORM:

```python
# Good - Using ORM
intersection = session.query(Intersection).filter_by(id=intersection_id).first()

# Bad - String concatenation
query = f"SELECT * FROM intersections WHERE id = '{intersection_id}'"
```

---

## Git Workflow

### Commit Messages

Follow **Conventional Commits**:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:

```
feat(rl): add hierarchical RL agent

Implement hierarchical RL for multi-intersection control
with improved coordination between agents.

Closes #123
```

```
fix(api): handle missing intersection gracefully

Return 404 instead of 500 when intersection not found.

Fixes #456
```

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

---

## Code Review Guidelines

### For Authors

- **Small PRs**: Keep pull requests focused and small
- **Clear Description**: Explain what and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs as needed
- **Self-Review**: Review your own code first

### For Reviewers

- **Be Constructive**: Provide helpful feedback
- **Check Tests**: Verify tests are adequate
- **Check Documentation**: Ensure docs are updated
- **Approve Promptly**: Don't block without good reason

---

## Performance Monitoring

### Logging Performance Metrics

```python
import time
from contextlib import contextmanager

@contextmanager
def time_operation(operation_name: str):
    """Context manager to time operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"{operation_name} took {duration:.3f}s")
        metrics.operation_duration.labels(operation=operation_name).observe(duration)

# Usage
with time_operation("decision_making"):
    decision = agent.select_action(state)
```

### Monitoring Best Practices

- **Track Key Metrics**: Decision latency, throughput, errors
- **Set Alerts**: Alert on anomalies
- **Dashboard**: Visualize metrics in real-time
- **Log Analysis**: Analyze logs for patterns

---

## Additional Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

---

**Last Updated**: November 30, 2024

