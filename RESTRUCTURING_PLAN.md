# Project Restructuring Plan

## Overview
This document outlines the restructuring of the adaptive-traffic monorepo into separate, focused projects.

## New Project Structure

### 1. **adaptive-traffic-core**
**Purpose**: Core traffic control system with RL, control strategies, environments, and forecasting

**Components**:
- `src/rl/` - Reinforcement Learning (DQN, training, inference)
- `src/env/` - Environments (TrafficEnv, SumoEnv, MarlEnv)
- `src/control/` - Control strategies (Fuzzy, Webster, GA, PSO)
- `src/forecast/` - Traffic forecasting (traffic_forecast.py, gnn_forecast.py)
- `src/optimization/` - Optimization algorithms (GA, PSO)
- `configs/` - Configuration files
- Training scripts (train_dqn.py, etc.)
- Demo scripts (demo.py, working_demo.py)

**Dependencies**: 
- adaptive-traffic-common (for shared utilities)
- adaptive-traffic-vision (optional, for video-based environments)

### 2. **adaptive-traffic-api**
**Purpose**: REST API and WebSocket service layer

**Components**:
- `src/api/` - All API code (routes, services, middleware, auth, etc.)
- `src/security/` - Security utilities
- API-specific tests
- API documentation

**Dependencies**:
- adaptive-traffic-core (for traffic control functionality)
- adaptive-traffic-common

### 3. **adaptive-traffic-research**
**Purpose**: Research platform for novel algorithms and experimental features

**Components**:
- `src/research/` - All research code
  - `novel_algorithms/` - Hierarchical RL, Imitation Learning, Model-based RL
  - `federated_learning/` - Federated learning components
  - `explainability/` - Explainability features
  - `benchmarking/` - Benchmark suites
  - `publication/` - Paper templates, reproducibility
- `src/experiment_tracking.py`
- Research-specific tests

**Dependencies**:
- adaptive-traffic-core
- adaptive-traffic-common

### 4. **adaptive-traffic-vision**
**Purpose**: Computer vision pipeline for vehicle detection

**Components**:
- `src/vision/` - All vision code (YOLO, video pipeline, ROI management)
- Vision-specific tests
- Model files (yolov8n.pt)

**Dependencies**:
- adaptive-traffic-common

### 5. **adaptive-traffic-deployment**
**Purpose**: Deployment infrastructure and configurations

**Components**:
- `deployment/` - Kubernetes, Helm, Docker configs
- `monitoring/` - Grafana dashboards, Prometheus configs
- Deployment scripts
- Deployment documentation

**Dependencies**:
- All other projects (deployment configs reference them)

### 6. **adaptive-traffic-common**
**Purpose**: Shared utilities and common code

**Components**:
- `src/utils/` - Common utilities (config, metrics, health, io, errors)
- `src/benchmarking/` - Public benchmarking utilities
- Shared data models
- Common test utilities

**Dependencies**: None (base library)

## Migration Strategy

### Phase 1: Create Structure
- ✅ Create new project directories
- Create base structure for each project

### Phase 2: Move Core Components
- Move `src/rl/` → `adaptive-traffic-core/src/rl/`
- Move `src/env/` → `adaptive-traffic-core/src/env/`
- Move `src/control/` → `adaptive-traffic-core/src/control/`
- Move `src/forecast/` → `adaptive-traffic-core/src/forecast/`
- Move `src/optimization/` → `adaptive-traffic-core/src/optimization/`
- Move `configs/` → `adaptive-traffic-core/configs/`
- Move training/demo scripts → `adaptive-traffic-core/`

### Phase 3: Move API Components
- Move `src/api/` → `adaptive-traffic-api/src/api/`
- Move `src/security/` → `adaptive-traffic-api/src/security/`
- Move API tests → `adaptive-traffic-api/tests/`

### Phase 4: Move Research Components
- Move `src/research/` → `adaptive-traffic-research/src/research/`
- Move research tests → `adaptive-traffic-research/tests/`

### Phase 5: Move Vision Components
- Move `src/vision/` → `adaptive-traffic-vision/src/vision/`
- Move vision tests → `adaptive-traffic-vision/tests/`
- Move model files → `adaptive-traffic-vision/models/`

### Phase 6: Move Deployment Components
- Move `deployment/` → `adaptive-traffic-deployment/deployment/`
- Move `monitoring/` → `adaptive-traffic-deployment/monitoring/`

### Phase 7: Create Common Library
- Move `src/utils/` → `adaptive-traffic-common/src/utils/`
- Move `src/benchmarking/` → `adaptive-traffic-common/src/benchmarking/`
- Extract shared code from other modules

### Phase 8: Update Imports
- Update all import statements to use new project structure
- Create proper package dependencies
- Update setup.py files for each project

### Phase 9: Create Project Documentation
- Create README.md for each project
- Create setup.py/pyproject.toml for each project
- Create requirements.txt for each project
- Update root README.md

## Import Path Changes

### Old Structure:
```python
from src.rl.dqn_agent import DQNAgent
from src.env.traffic_env import TrafficEnv
from src.vision import YOLOQueueEstimator
```

### New Structure:
```python
# In adaptive-traffic-core
from adaptive_traffic_core.rl.dqn_agent import DQNAgent
from adaptive_traffic_core.env.traffic_env import TrafficEnv
from adaptive_traffic_common.utils.config import load_config

# In adaptive-traffic-vision
from adaptive_traffic_vision.vision import YOLOQueueEstimator
from adaptive_traffic_common.utils.io import save_image
```

## Dependency Management

Each project will have its own:
- `requirements.txt` - Project-specific dependencies
- `setup.py` or `pyproject.toml` - Package configuration
- `README.md` - Project documentation
- `.gitignore` - Project-specific ignores

## Testing Strategy

- Each project maintains its own test suite
- Integration tests can be in a separate location or in dependent projects
- Common test utilities in `adaptive-traffic-common`

## Next Steps

1. Complete directory structure creation
2. Move files systematically
3. Update imports
4. Create package configurations
5. Test each project independently
6. Update documentation

