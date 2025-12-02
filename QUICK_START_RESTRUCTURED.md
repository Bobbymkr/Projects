# Quick Start Guide - Restructured Project

## 🎯 Overview

The project has been restructured into 6 focused sub-projects. This guide helps you get started quickly.

## 📦 Installation Order

Install packages in this order (dependencies first):

```bash
# 1. Base library (no dependencies)
cd adaptive-traffic-common
pip install -e .

# 2. Core system (depends on common)
cd ../adaptive-traffic-core
pip install -e .

# 3. Vision (depends on common)
cd ../adaptive-traffic-vision
pip install -e .

# 4. API (depends on core, common)
cd ../adaptive-traffic-api
pip install -e .

# 5. Research (depends on core, common)
cd ../adaptive-traffic-research
pip install -e .
```

Or install all at once from root:

```bash
pip install -e adaptive-traffic-common
pip install -e adaptive-traffic-core
pip install -e adaptive-traffic-vision
pip install -e adaptive-traffic-api
pip install -e adaptive-traffic-research
```

## 🚀 Quick Examples

### Core - Train a DQN Agent

```python
from adaptive_traffic_core.rl.dqn_agent import DQNAgent, DQNConfig
from adaptive_traffic_core.env.traffic_env import TrafficEnv

env = TrafficEnv(config_path="adaptive-traffic-core/configs/intersection.json")
config = DQNConfig()
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)

# Training loop
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

### Vision - Detect Vehicles

```python
from adaptive_traffic_vision.vision import YOLOQueueEstimator, VideoInputStream

video = VideoInputStream(source=0)  # Webcam
estimator = YOLOQueueEstimator(model_path="adaptive-traffic-vision/models/yolov8n.pt")

for frame in video:
    queues = estimator.estimate_queues(frame)
    print(f"Queue lengths: {queues}")
```

### API - Start Server

```bash
cd adaptive-traffic-api
python -m adaptive_traffic_api.api.main
# Or
python scripts/start_api.py
```

Visit `http://localhost:8000/docs` for API documentation.

## 📁 Project Locations

| Project | Location | Purpose |
|---------|----------|---------|
| **Core** | `adaptive-traffic-core/` | RL, control strategies, environments |
| **API** | `adaptive-traffic-api/` | REST API, WebSocket, GraphQL |
| **Research** | `adaptive-traffic-research/` | Novel algorithms, federated learning |
| **Vision** | `adaptive-traffic-vision/` | YOLOv8, vehicle detection |
| **Deployment** | `adaptive-traffic-deployment/` | K8s, Helm, Docker configs |
| **Common** | `adaptive-traffic-common/` | Shared utilities |

## 🔄 Import Changes

**Old way:**
```python
from src.rl.dqn_agent import DQNAgent
```

**New way:**
```python
from adaptive_traffic_core.rl.dqn_agent import DQNAgent
```

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for complete mapping.

## 🧪 Testing

Test each project:

```bash
# Test core
cd adaptive-traffic-core
pytest tests/

# Test API
cd ../adaptive-traffic-api
pytest tests/

# Test vision
cd ../adaptive-traffic-vision
pytest tests/
```

## 📚 Documentation

- [Main README](./README_RESTRUCTURED.md) - Overview and structure
- [Restructuring Plan](./RESTRUCTURING_PLAN.md) - Detailed plan
- [Migration Guide](./MIGRATION_GUIDE.md) - How to update imports
- [Restructuring Summary](./RESTRUCTURING_SUMMARY.md) - What was done

Each project has its own README:
- [Core README](./adaptive-traffic-core/README.md)
- [API README](./adaptive-traffic-api/README.md)
- [Research README](./adaptive-traffic-research/README.md)
- [Vision README](./adaptive-traffic-vision/README.md)
- [Deployment README](./adaptive-traffic-deployment/README.md)
- [Common README](./adaptive-traffic-common/README.md)

## ⚠️ Important Notes

1. **Install order matters**: Install `adaptive-traffic-common` first
2. **Import paths changed**: Use new package names (see MIGRATION_GUIDE.md)
3. **Config paths**: Configs are in `adaptive-traffic-core/configs/`
4. **Original files**: Original files still exist - you can remove them after verification

## 🆘 Troubleshooting

### ModuleNotFoundError
**Solution:** Make sure you've installed all required packages in the correct order.

### Import errors
**Solution:** Check that you're using the new import paths from MIGRATION_GUIDE.md.

### Config file not found
**Solution:** Update paths to point to `adaptive-traffic-core/configs/`

## ✅ Verification Checklist

- [ ] All packages installed successfully
- [ ] Can import from each package
- [ ] Core training script works
- [ ] API server starts
- [ ] Vision pipeline works
- [ ] Tests pass (if applicable)

## 🎉 You're Ready!

The project is now restructured and ready to use. Each sub-project can be developed independently while sharing common utilities through `adaptive-traffic-common`.

