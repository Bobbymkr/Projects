# Quick Start Guide - Phase 0 Optimizations

This guide shows you how to quickly start using the Phase 0 optimizations from OPTIMIZATION_ROADMAP.md.

## 🚀 Quick Start

### 1. Enhanced Reward Function (Automatic)

The enhanced reward function is now the default in `TrafficEnv`. No changes needed!

```python
from src.env.traffic_env import TrafficEnv

config = {
    "num_lanes": 4,
    "phase_lanes": [[0, 1], [2, 3]],
    # ... other config ...
    # Enhanced reward weights (optional - these are defaults)
    "reward_weights": {
        "queue": -0.4,
        "wait_penalty": -0.3,
        "throughput": 0.2,
        "efficiency": 0.1,
        "queue_reduction": 0.05,
        "safety": -1000.0
    }
}

env = TrafficEnv(config=config)
# Reward function is automatically enhanced!
```

### 2. Training with Stability Framework

```python
from src.rl.training_stability import TrainingStabilityFramework
import torch.optim as optim

# Setup your agent (example with DQN)
optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-3)

# Create stability framework
stability_config = {
    "grad_clip_norm": 10.0,
    "lr_scheduler": {
        "enabled": True,
        "type": "cosine",
        "params": {"T_max": 2000, "eta_min": 1e-6}
    },
    "target_update_tau": 0.005,
    "use_soft_update": True,
    "exploration": {
        "enabled": True,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "decay_type": "linear",
        "decay_steps": 10000
    }
}

framework = TrainingStabilityFramework(
    optimizer=optimizer,
    policy_net=agent.policy_net,
    target_net=agent.target_net,
    config=stability_config
)

# In your training loop:
for episode in range(num_episodes):
    # ... training code ...
    
    # Apply stability techniques
    framework.clip_gradients()
    framework.update_target_network()
    framework.step_scheduler(metric=episode_reward)
    epsilon = framework.get_exploration_rate()
```

### 3. Convergence Detection

```python
from src.rl.convergence_monitor import ConvergenceMonitor

# Create monitor
monitor = ConvergenceMonitor(
    window=100,        # Moving average window
    threshold=0.01,    # Improvement threshold
    patience=500,       # Episodes without improvement
    min_episodes=200,  # Minimum episodes before stopping
    mode="maximize"     # or "minimize"
)

# In training loop:
for episode in range(num_episodes):
    # ... training ...
    
    status = monitor.update(episode_reward, episode)
    
    if status["should_stop"]:
        print(f"Early stopping at episode {episode}")
        print(f"Best reward: {status['best_reward']:.2f}")
        break
```

### 4. Complete Example (All Together)

Use the integrated training script:

```bash
python scripts/train_with_optimization.py \
    --config configs/intersection.json \
    --episodes 2000 \
    --output ./runs/optimized_training \
    --agent DQN
```

Or integrate manually:

```python
from src.env.traffic_env import TrafficEnv
from src.rl.training_stability import TrainingStabilityFramework
from src.rl.convergence_monitor import ConvergenceMonitor
from scripts.train_with_optimization import train_with_optimizations

# Load config and create environment
with open("configs/intersection.json", 'r') as f:
    config = json.load(f)

env = TrafficEnv(config=config)

# Create agent (your agent here)
agent = YourAgent(...)

# Configure optimizations
convergence_config = {
    "window": 100,
    "threshold": 0.01,
    "patience": 500,
    "min_episodes": 200
}

stability_config = {
    "grad_clip_norm": 10.0,
    "lr_scheduler": {"enabled": True, "type": "cosine"},
    "target_update_tau": 0.005,
    "use_soft_update": True,
    "exploration": {"enabled": True, "decay_type": "linear"}
}

# Train with all optimizations
results = train_with_optimizations(
    agent=agent,
    env=env,
    episodes=2000,
    convergence_config=convergence_config,
    stability_config=stability_config,
    output_dir=Path("./runs/optimized")
)
```

## 📊 Expected Results

After implementing Phase 0:

- **Performance:** 15-20% improvement (avg reward: -107.81 → -95 to -100)
- **Stability:** 67% variance reduction (std: 6.0-6.5 → < 2.0)
- **Speed:** 30-40% faster training (early stopping)

## 🔧 Configuration Options

### Reward Weights
Adjust reward component weights in environment config:
```python
"reward_weights": {
    "queue": -0.4,           # Queue length penalty
    "wait_penalty": -0.3,     # Wait time penalty
    "throughput": 0.2,        # Throughput bonus
    "efficiency": 0.1,        # Action efficiency
    "queue_reduction": 0.05,  # Queue reduction bonus
    "safety": -1000.0         # Safety violations
}
```

### Stability Framework
```python
stability_config = {
    "grad_clip_norm": 10.0,    # Gradient clipping norm
    "grad_clip_value": None,   # Or use value clipping
    "lr_scheduler": {
        "enabled": True,
        "type": "cosine",      # or "cosine_warm_restart", "plateau"
        "params": {...}
    },
    "target_update_tau": 0.005,  # Soft update coefficient
    "use_soft_update": True,
    "exploration": {
        "enabled": True,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "decay_type": "linear",  # or "exponential", "cosine"
        "decay_steps": 10000
    }
}
```

### Convergence Monitor
```python
monitor = ConvergenceMonitor(
    window=100,        # Episodes for moving average
    threshold=0.01,    # Minimum improvement
    patience=500,      # Episodes without improvement
    min_episodes=200,  # Minimum training episodes
    mode="maximize"    # or "minimize"
)
```

## 📚 More Information

- **Full Documentation:** See `PHASE_0_IMPLEMENTATION_SUMMARY.md`
- **Roadmap:** See `OPTIMIZATION_ROADMAP.md` for complete strategy
- **Code:** 
  - `src/env/traffic_env.py` - Enhanced reward function
  - `src/rl/training_stability.py` - Stability framework
  - `src/rl/convergence_monitor.py` - Convergence detection
  - `scripts/train_with_optimization.py` - Complete example

## ✅ Next Steps

After Phase 0 is working:

1. **Phase 1:** Hyperparameter Optimization
2. **Phase 2:** Advanced Training Techniques (PER, Curriculum Learning)
3. **Phase 3:** Architecture Enhancements (GNN, Enhanced Transformer)

See `OPTIMIZATION_ROADMAP.md` for the complete roadmap.

