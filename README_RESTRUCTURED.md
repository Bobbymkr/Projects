# Adaptive Traffic Signal Control System

> **World-class intelligent traffic signal control system using Deep Reinforcement Learning, Computer Vision, and Multi-Agent coordination for optimizing urban traffic flow.**

## 🎯 Project Restructuring

This project has been restructured into separate, focused sub-projects for better organization and maintainability:

### 📦 Sub-Projects

1. **[adaptive-traffic-core](./adaptive-traffic-core/)** - Core traffic control system
   - Reinforcement Learning (DQN, training, inference)
   - Control strategies (Fuzzy Logic, Webster, GA, PSO)
   - Traffic environments (TrafficEnv, SumoEnv, MarlEnv)
   - Traffic forecasting (CNN-LSTM, GNN)

2. **[adaptive-traffic-api](./adaptive-traffic-api/)** - REST API and WebSocket service
   - FastAPI-based REST endpoints
   - WebSocket support for real-time updates
   - GraphQL API
   - Authentication and security

3. **[adaptive-traffic-research](./adaptive-traffic-research/)** - Research platform
   - Novel algorithms (Hierarchical RL, Imitation Learning)
   - Federated learning
   - Explainability tools
   - Benchmarking suites

4. **[adaptive-traffic-vision](./adaptive-traffic-vision/)** - Computer vision pipeline
   - YOLOv8-based vehicle detection
   - Queue length estimation
   - Video processing pipeline

5. **[adaptive-traffic-deployment](./adaptive-traffic-deployment/)** - Deployment infrastructure
   - Kubernetes manifests
   - Helm charts
   - Docker configurations
   - Monitoring (Grafana, Prometheus)

6. **[adaptive-traffic-common](./adaptive-traffic-common/)** - Shared utilities
   - Common utilities and helpers
   - Shared data models
   - Benchmarking utilities

## 🚀 Quick Start

### Installation

```bash
# Install all projects (from root directory)
pip install -e adaptive-traffic-common
pip install -e adaptive-traffic-core
pip install -e adaptive-traffic-api
pip install -e adaptive-traffic-vision
pip install -e adaptive-traffic-research
```

### Basic Usage

```python
# Core functionality
from adaptive_traffic_core.rl.dqn_agent import DQNAgent, DQNConfig
from adaptive_traffic_core.env.traffic_env import TrafficEnv

# Vision
from adaptive_traffic_vision.vision import YOLOQueueEstimator

# API (if running API server)
# Access via http://localhost:8000/docs
```

## 📁 Project Structure

```
adaptive_traffic/
├── adaptive-traffic-core/          # Core traffic control
│   ├── src/
│   │   ├── rl/                     # Reinforcement Learning
│   │   ├── env/                    # Environments
│   │   ├── control/                # Control strategies
│   │   ├── forecast/               # Traffic forecasting
│   │   └── optimization/           # Optimization algorithms
│   ├── configs/                    # Configuration files
│   └── README.md
│
├── adaptive-traffic-api/            # API service layer
│   ├── src/
│   │   ├── api/                    # API code
│   │   └── security/               # Security utilities
│   └── README.md
│
├── adaptive-traffic-research/       # Research platform
│   ├── src/
│   │   └── research/               # Research components
│   └── README.md
│
├── adaptive-traffic-vision/         # Computer vision
│   ├── src/
│   │   └── vision/                 # Vision code
│   ├── models/                     # Model files
│   └── README.md
│
├── adaptive-traffic-deployment/     # Deployment
│   ├── deployment/                 # K8s, Helm, Docker
│   ├── monitoring/                 # Monitoring configs
│   └── README.md
│
├── adaptive-traffic-common/         # Shared utilities
│   ├── src/
│   │   ├── utils/                  # Utilities
│   │   └── benchmarking/           # Benchmarking
│   └── README.md
│
└── README_RESTRUCTURED.md          # This file
```

## 🔗 Dependencies

```
adaptive-traffic-common (base)
    ↑
    ├── adaptive-traffic-core
    │   └── adaptive-traffic-api
    │
    ├── adaptive-traffic-vision
    │   └── adaptive-traffic-core (optional)
    │
    └── adaptive-traffic-research
        └── adaptive-traffic-core
```

## 📚 Documentation

Each sub-project has its own README with detailed documentation:

- [Core Documentation](./adaptive-traffic-core/README.md)
- [API Documentation](./adaptive-traffic-api/README.md)
- [Research Documentation](./adaptive-traffic-research/README.md)
- [Vision Documentation](./adaptive-traffic-vision/README.md)
- [Deployment Documentation](./adaptive-traffic-deployment/README.md)
- [Common Documentation](./adaptive-traffic-common/README.md)

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd adaptive_traffic

# Install all projects in development mode
pip install -e adaptive-traffic-common
pip install -e adaptive-traffic-core
pip install -e adaptive-traffic-api
pip install -e adaptive-traffic-vision
pip install -e adaptive-traffic-research
```

### Running Tests

Each project has its own test suite. Run tests from each project directory:

```bash
cd adaptive-traffic-core
pytest tests/

cd ../adaptive-traffic-api
pytest tests/
```

## 🔄 Migration from Old Structure

If you're migrating from the old monolithic structure, see [RESTRUCTURING_PLAN.md](./RESTRUCTURING_PLAN.md) for details.

### Import Changes

**Old:**
```python
from src.rl.dqn_agent import DQNAgent
from src.env.traffic_env import TrafficEnv
```

**New:**
```python
from adaptive_traffic_core.rl.dqn_agent import DQNAgent
from adaptive_traffic_core.env.traffic_env import TrafficEnv
```

## 📊 Performance Results

| **Algorithm** | **Wait Time** | **Queue Length** | **Efficiency** | **Grade** |
|---------------|---------------|------------------|----------------|-----------|
| **Fuzzy Control** | 8.51s | 12.5 vehicles | 1.2123 | **A+** |
| **GNN Forecasting** | 13.58s | 15.4 vehicles | 1.1848 | **A** |
| **DQN (6000 episodes)** | 21.47s | 23.4 vehicles | 1.2064 | **B+** |
| **Traditional (Webster)** | 27.37s | 24.9 vehicles | 1.1612 | **C** |

## 🤝 Contributing

We welcome contributions! Each sub-project can be developed independently. See individual project READMEs for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SUMO**: Simulation of Urban MObility
- **YOLOv8**: Ultralytics object detection
- **TensorFlow/PyTorch**: Deep learning frameworks
- **OpenCV**: Computer vision library

---

<div align="center">

**Star this repo if you find it useful!**

**🚦 Building smarter cities with AI 🚦**

</div>

