# Advanced Training Techniques Guide

This guide covers the advanced training techniques available in the Adaptive Traffic Control System: Prioritized Experience Replay (PER), Distributional RL, and Curriculum Learning.

## Table of Contents

1. [Prioritized Experience Replay (PER)](#prioritized-experience-replay-per)
2. [Distributional RL](#distributional-rl)
3. [Curriculum Learning](#curriculum-learning)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Best Practices](#best-practices)

---

## Prioritized Experience Replay (PER)

### Overview

Prioritized Experience Replay improves sample efficiency by prioritizing important transitions (those with high TD-error) for replay. This typically results in 15-20% better sample efficiency.

### Features

- TD-error based prioritization (α=0.6)
- Importance sampling correction (β=0.4 to 1.0, annealing)
- Automatic priority updates after training
- Backward compatible (can disable to use uniform replay)

### Usage

#### Enable PER via Environment Variables

```bash
# Enable PER
export ADAPTIVE_TRAFFIC_USE_PER=1
export ADAPTIVE_TRAFFIC_PER_ALPHA=0.6
export ADAPTIVE_TRAFFIC_PER_BETA=0.4
export ADAPTIVE_TRAFFIC_PER_BETA_INC=0.001

# Run training
python src/rl/train_dqn_simple.py \
    --config configs/intersection.json \
    --episodes 500 \
    --out runs/dqn_with_per
```

#### Enable PER Programmatically

```python
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.env.traffic_env import TrafficEnv

env = TrafficEnv(config_path="configs/intersection.json")

# Create config with PER enabled
cfg = DQNConfig()
cfg.use_per = True
cfg.per_alpha = 0.6  # Priority exponent (0=uniform, 1=fully prioritized)
cfg.per_beta = 0.4   # Importance sampling exponent (start value)
cfg.per_beta_increment = 0.001  # Beta annealing rate

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    cfg=cfg
)

# Training loop (same as normal DQN)
for episode in range(500):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.push(obs, action, reward, next_obs, done or truncated)
        loss = agent.train_step()  # PER priorities updated automatically
        obs = next_obs
```

### Parameters

- **`use_per`**: Enable/disable PER (default: `False`)
- **`per_alpha`**: Priority exponent (default: `0.6`)
  - `0.0` = uniform sampling (no prioritization)
  - `1.0` = full prioritization
  - Recommended: `0.6`
- **`per_beta`**: Importance sampling exponent start value (default: `0.4`)
  - Starts at `0.4` and anneals to `1.0`
  - Corrects for bias introduced by prioritization
- **`per_beta_increment`**: Beta annealing rate (default: `0.001`)
  - Amount to increment beta per sample

### When to Use PER

- ✅ When sample efficiency is important
- ✅ When training time is limited
- ✅ For complex environments with sparse rewards
- ❌ When training speed is more important than sample efficiency
- ❌ For very simple environments

### Performance Impact

Expected improvements:
- **Sample Efficiency**: 15-20% improvement
- **Training Speed**: 5-10% slower (due to priority updates)
- **Final Performance**: Similar or slightly better

---

## Distributional RL

### Overview

Distributional RL models the full distribution of returns rather than just the expected value. This provides better uncertainty estimation and more stable learning. Available algorithms: C51 and QR-DQN.

### Features

- **C51**: 51-atom categorical distribution
- **QR-DQN**: Quantile regression with 200 quantiles
- Better uncertainty estimation
- Risk-aware decision making
- More stable learning

### Usage

#### Training C51 Agent

```bash
python src/rl/train_distributional_dqn.py \
    --config configs/intersection.json \
    --episodes 500 \
    --algorithm C51 \
    --out runs/c51_training
```

#### Training QR-DQN Agent

```bash
python src/rl/train_distributional_dqn.py \
    --config configs/intersection.json \
    --episodes 500 \
    --algorithm QR-DQN \
    --out runs/qrdqn_training
```

#### With Curriculum Learning

```bash
# Curriculum learning is enabled by default
python src/rl/train_distributional_dqn.py \
    --config configs/intersection.json \
    --episodes 500 \
    --algorithm C51

# Disable curriculum learning
python src/rl/train_distributional_dqn.py \
    --config configs/intersection.json \
    --episodes 500 \
    --algorithm C51 \
    --no-curriculum
```

#### Programmatic Usage

```python
from src.rl.distributional_rl import DistributionalDQNAgent, DistributionalRLConfig
from src.env.traffic_env import TrafficEnv
import torch

env = TrafficEnv(config_path="configs/intersection.json")

# Create distributional RL config
config = DistributionalRLConfig(
    algorithm="C51",  # or "QR-DQN"
    num_atoms=51,  # For C51
    num_quantiles=200,  # For QR-DQN
    v_min=-10.0,
    v_max=10.0,
    risk_type="neutral",  # "neutral", "risk_averse", "risk_seeking"
)

# Create agent
agent = DistributionalDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Training loop
for episode in range(500):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.push(obs, action, reward, next_obs, done or truncated)
        loss = agent.train_step()
        obs = next_obs
```

### Algorithm Comparison

| Algorithm | Atoms/Quantiles | Best For | Performance |
|-----------|----------------|----------|-------------|
| **C51** | 51 atoms | General use, good balance | 10-15% improvement |
| **QR-DQN** | 200 quantiles | Better uncertainty | 15-20% improvement |

### When to Use Distributional RL

- ✅ When uncertainty estimation is important
- ✅ For risk-aware decision making
- ✅ When learning stability is a concern
- ✅ For complex reward distributions
- ❌ When training speed is critical (slower than standard DQN)
- ❌ For very simple environments

### Requirements

- PyTorch (required)
- CUDA (optional, for GPU acceleration)

```bash
pip install torch
```

---

## Curriculum Learning

### Overview

Curriculum Learning progressively increases difficulty during training, starting with easy scenarios and gradually moving to harder ones. This typically results in 20-25% faster convergence.

### Features

- 6 progressive difficulty levels (Very Easy → Extreme)
- Adaptive progression based on performance
- Automatic difficulty adjustment
- Performance threshold-based advancement

### Usage

#### Curriculum Learning is Enabled by Default

The training script `src/rl/train_dqn_simple.py` includes curriculum learning by default:

```bash
python src/rl/train_dqn_simple.py \
    --config configs/intersection.json \
    --episodes 500 \
    --out runs/dqn_with_curriculum
```

#### Programmatic Usage

```python
from src.rl.curriculum_learning import TrafficCurriculum
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.env.traffic_env import TrafficEnv

env = TrafficEnv(config_path="configs/intersection.json")

# Initialize curriculum
base_arrival_rates = [0.3, 0.3, 0.3, 0.3]  # Base traffic rates
curriculum = TrafficCurriculum(
    base_arrival_rates=base_arrival_rates,
    performance_threshold=0.7,  # Performance needed to advance
    min_episodes_per_level=50,  # Minimum episodes before advancing
    performance_window=100,  # Episodes to evaluate performance
)

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    cfg=DQNConfig()
)

# Training loop with curriculum
for episode in range(500):
    # Update environment with current curriculum level
    env.arrival_rates = curriculum.get_arrival_rates()
    
    obs, info = env.reset()
    episode_reward = 0.0
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.push(obs, action, reward, next_obs, done or truncated)
        agent.train_step()
        episode_reward += reward
        obs = next_obs
    
    # Update curriculum based on performance
    curriculum.update_performance(episode_reward, episode)
    
    # Check current level
    current_level = curriculum.get_current_level()
    print(f"Episode {episode}: Level {current_level.level_id} - {current_level.description}")
```

### Curriculum Levels

1. **Level 0 - Very Easy**: 10% traffic density (light traffic)
2. **Level 1 - Easy**: 30% traffic density (moderate traffic)
3. **Level 2 - Medium**: 50% traffic density (normal traffic)
4. **Level 3 - Hard**: 70% traffic density (heavy traffic)
5. **Level 4 - Very Hard**: 90% traffic density (very heavy traffic)
6. **Level 5 - Extreme**: 100% traffic density (rush hour)

### Parameters

- **`base_arrival_rates`**: Base traffic arrival rates per lane
- **`performance_threshold`**: Performance ratio needed to advance (default: `0.7`)
- **`min_episodes_per_level`**: Minimum episodes before allowing advancement (default: `50`)
- **`performance_window`**: Number of recent episodes to evaluate (default: `100`)

### When to Use Curriculum Learning

- ✅ For faster convergence
- ✅ When starting from scratch
- ✅ For complex environments
- ✅ When sample efficiency matters
- ❌ When fine-tuning pre-trained models
- ❌ For very simple environments

### Performance Impact

Expected improvements:
- **Convergence Speed**: 20-25% faster
- **Final Performance**: Similar or slightly better
- **Training Stability**: More stable learning curve

---

## Performance Benchmarking

### Benchmark PER vs Uniform Replay

```bash
python scripts/benchmark_performance.py \
    --config configs/intersection.json \
    --episodes 200 \
    --benchmark per \
    --out runs/per_benchmark
```

### Benchmark Curriculum Learning

```bash
python scripts/benchmark_performance.py \
    --config configs/intersection.json \
    --episodes 200 \
    --benchmark curriculum \
    --out runs/curriculum_benchmark
```

### Validate Curriculum Learning

```bash
python scripts/validate_curriculum.py \
    --config configs/intersection.json \
    --episodes 300 \
    --runs 3 \
    --out runs/curriculum_validation
```

This runs multiple independent training runs and compares performance with and without curriculum learning, providing statistical significance.

---

## Best Practices

### Combining Techniques

You can combine multiple techniques for maximum benefit:

```python
# PER + Curriculum Learning
cfg = DQNConfig()
cfg.use_per = True
cfg.per_alpha = 0.6
cfg.per_beta = 0.4

agent = DQNAgent(state_dim, action_dim, cfg=cfg)
curriculum = TrafficCurriculum(base_arrival_rates, ...)

# Training with both
for episode in range(500):
    env.arrival_rates = curriculum.get_arrival_rates()
    # ... training loop ...
    curriculum.update_performance(episode_reward, episode)
```

### Recommended Configurations

#### For Fast Training (Sample Efficiency)
- ✅ PER enabled (α=0.6, β=0.4)
- ✅ Curriculum Learning enabled
- ⚠️ Distributional RL (slower but better)

#### For Best Performance
- ✅ Distributional RL (C51 or QR-DQN)
- ✅ PER enabled
- ✅ Curriculum Learning enabled

#### For Simple/Quick Experiments
- ❌ PER (adds complexity)
- ❌ Curriculum Learning (adds complexity)
- ❌ Distributional RL (requires PyTorch)

### Hyperparameter Tuning

Use the hyperparameter optimization script:

```bash
python scripts/optimize_hyperparameters.py \
    --config configs/intersection.json \
    --trials 100 \
    --episodes 50 \
    --out runs/hyperopt
```

This will optimize learning rate, batch size, gamma, epsilon, and PER parameters.

---

## Troubleshooting

### PER Not Improving Performance

- Try adjusting `per_alpha` (lower = more uniform, higher = more prioritized)
- Ensure `per_beta` anneals properly (check `per_beta_increment`)
- Verify importance sampling weights are being applied

### Curriculum Learning Not Progressing

- Check `performance_threshold` (may be too high)
- Reduce `min_episodes_per_level` for faster progression
- Verify performance metrics are improving

### Distributional RL Training Fails

- Ensure PyTorch is installed: `pip install torch`
- Check CUDA availability if using GPU
- Verify state/action dimensions match environment

---

## References

- **PER**: Schaul et al. "Prioritized Experience Replay" (ICLR 2016)
- **C51**: Bellemare et al. "A Distributional Perspective on Reinforcement Learning" (ICML 2017)
- **QR-DQN**: Dabney et al. "Distributional Reinforcement Learning with Quantile Regression" (AAAI 2018)
- **Curriculum Learning**: Bengio et al. "Curriculum Learning" (ICML 2009)

---

## Support

For issues or questions:
1. Check test files: `tests/unit/research/test_*.py`
2. Review implementation: `src/rl/prioritized_replay.py`, `src/rl/distributional_rl.py`, `src/rl/curriculum_learning.py`
3. Run validation scripts to verify functionality

