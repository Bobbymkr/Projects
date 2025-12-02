# Adaptive Traffic Common

Shared utilities and common code for all Adaptive Traffic projects.

## Overview

This package provides shared functionality used across all Adaptive Traffic projects:

- **Utilities**: Configuration, metrics, health checks, I/O operations
- **Benchmarking**: Public benchmarking utilities
- **Common Models**: Shared data models and types
- **Test Utilities**: Common testing helpers

## Installation

```bash
pip install -e .
```

## Usage

```python
from adaptive_traffic_common.utils.config import load_config
from adaptive_traffic_common.utils.metrics import calculate_metrics
from adaptive_traffic_common.utils.health import health_check

# Load configuration
config = load_config("config.json")

# Calculate metrics
metrics = calculate_metrics(data)

# Health check
status = health_check()
```

## Project Structure

```
adaptive-traffic-common/
├── src/
│   ├── utils/            # Utility functions
│   └── benchmarking/      # Benchmarking utilities
└── README.md
```

## Dependencies

This is a base library with minimal dependencies:
- `numpy>=1.26.0`
- `pydantic>=2.7.0`

## License

MIT License

