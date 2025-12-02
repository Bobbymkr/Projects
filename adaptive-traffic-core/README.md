# Adaptive Traffic Core

The core traffic signal control system using Deep Reinforcement Learning, control strategies, and traffic forecasting.

## Overview

This package provides the foundational components for intelligent traffic signal control:

- **Reinforcement Learning**: DQN agents for learning optimal signal timing
- **Control Strategies**: Fuzzy Logic, Webster's Method, Genetic Algorithms, PSO
- **Environments**: Traffic simulation environments (TrafficEnv, SumoEnv, MarlEnv)
- **Traffic Forecasting**: CNN-LSTM and GNN-based traffic prediction models
- **Optimization**: Genetic Algorithms and Particle Swarm Optimization

## Installation

```bash
pip install -e .
```

## Dependencies

- `adaptive-traffic-common` - Shared utilities
- `adaptive-traffic-vision` (optional) - For video-based environments

## Quick Start

```python
from adaptive_traffic_core.rl.dqn_agent import DQNAgent, DQNConfig
from adaptive_traffic_core.env.traffic_env import TrafficEnv

# Create environment
env = TrafficEnv(config_path="configs/intersection.json")

# Create agent
config = DQNConfig()
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)

# Train
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
```

## Training

```bash
# Basic training
python train_dqn.py --episodes 100 --config configs/intersection.json

# With SUMO
python train_dqn.py --episodes 100 --use_sumo --config configs/grid.sumocfg

# Multi-agent training
python train_dqn.py --episodes 100 --marl --config configs/grid.sumocfg
```

## Project Structure

```
adaptive-traffic-core/
├── src/
│   ├── rl/              # Reinforcement Learning
│   ├── env/              # Environments
│   ├── control/          # Control strategies
│   ├── forecast/         # Traffic forecasting
│   └── optimization/     # Optimization algorithms
├── configs/              # Configuration files
├── train_*.py            # Training scripts
├── demo*.py              # Demo scripts
└── README.md
```

## Documentation

See the main project README for comprehensive documentation.

## License

MIT License

