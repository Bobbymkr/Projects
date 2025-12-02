# Adaptive Traffic Research

Research platform for novel algorithms and experimental features in traffic signal control.

## Overview

This package provides research-oriented components:

- **Novel Algorithms**: Hierarchical RL, Imitation Learning, Model-based RL
- **Federated Learning**: Privacy-preserving distributed learning
- **Explainability**: Model interpretability and explanation tools
- **Benchmarking**: Comprehensive benchmark suites
- **Publication Tools**: Paper templates and reproducibility tools

## Installation

```bash
pip install -e .
```

## Dependencies

- `adaptive-traffic-core` - Core traffic control functionality
- `adaptive-traffic-common` - Shared utilities

## Quick Start

```python
from adaptive_traffic_research.research.novel_algorithms.hierarchical_rl import HierarchicalRL
from adaptive_traffic_core.env.traffic_env import TrafficEnv

# Create environment
env = TrafficEnv(config_path="configs/intersection.json")

# Create hierarchical RL agent
agent = HierarchicalRL(env)

# Train
agent.train(episodes=1000)
```

## Research Areas

### Novel Algorithms
- Hierarchical Reinforcement Learning
- Imitation Learning
- Model-based Reinforcement Learning

### Federated Learning
- Privacy-preserving distributed training
- Secure aggregation mechanisms

### Explainability
- Model interpretability
- Decision explanation tools

### Benchmarking
- Performance benchmarks
- Algorithm comparison tools

## Project Structure

```
adaptive-traffic-research/
├── src/
│   └── research/
│       ├── novel_algorithms/    # Novel RL algorithms
│       ├── federated_learning/  # Federated learning
│       ├── explainability/      # Explainability tools
│       ├── benchmarking/        # Benchmark suites
│       └── publication/         # Publication tools
└── README.md
```

## License

MIT License

