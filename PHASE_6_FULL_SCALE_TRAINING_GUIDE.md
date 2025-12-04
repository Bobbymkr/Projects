# Phase 6 Full-Scale Training Experiments Guide

## Overview

This guide provides instructions for running full-scale training experiments for Phase 6 Advanced RL Algorithms (PPO, SAC, Rainbow DQN).

## Quick Start

### Basic Training (All Algorithms)

```bash
python scripts/train_phase6_full_scale.py --episodes 1000 --output ./runs/phase6_full_scale
```

### Train Single Algorithm

```bash
# Train PPO only
python scripts/train_phase6_full_scale.py --algorithm PPO --episodes 1000

# Train SAC only
python scripts/train_phase6_full_scale.py --algorithm SAC --episodes 1000

# Train Rainbow DQN only
python scripts/train_phase6_full_scale.py --algorithm "Rainbow DQN" --episodes 1000
```

### Using Optimized Hyperparameters

First, run hyperparameter optimization:

```bash
# Optimize PPO
python scripts/optimize_phase6.py --algorithm PPO --trials 50 --episodes 100

# Optimize SAC
python scripts/optimize_phase6.py --algorithm SAC --trials 50 --episodes 100

# Optimize Rainbow DQN
python scripts/optimize_phase6.py --algorithm "Rainbow DQN" --trials 50 --episodes 100
```

Then use optimized parameters:

```bash
python scripts/train_phase6_full_scale.py \
    --episodes 1000 \
    --use-optimized \
    --optimized-params ./runs/phase6_optimization/ppo_best_params.json
```

## Command Line Arguments

### Required Arguments
- None (all have defaults)

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `configs/intersection.json` | Environment config file |
| `--episodes` | int | `1000` | Number of training episodes per algorithm |
| `--output` | str | `./runs/phase6_full_scale` | Output directory for results |
| `--eval-interval` | int | `100` | Episodes between evaluations |
| `--save-interval` | int | `200` | Episodes between model checkpoints |
| `--use-optimized` | flag | `False` | Use optimized hyperparameters |
| `--optimized-params` | str | `None` | Path to optimized parameters JSON |
| `--algorithm` | str | `all` | Algorithm to train: `PPO`, `SAC`, `Rainbow DQN`, or `all` |

## Output Structure

```
runs/phase6_full_scale/
├── models/
│   ├── ppo/
│   │   ├── policy_net_ep200.pt
│   │   ├── value_net_ep200.pt
│   │   └── ...
│   ├── sac/
│   │   ├── actor_ep200.pt
│   │   ├── critic1_ep200.pt
│   │   └── ...
│   └── rainbow_dqn/
│       ├── q_net_ep200.pt
│       └── ...
├── logs/
│   └── (training logs)
├── plots/
│   └── (visualization plots)
├── training_results.json      # Complete training results
└── training_summary.txt        # Human-readable summary
```

## Training Results Format

The `training_results.json` file contains:

```json
{
  "PPO": {
    "algorithm": "PPO",
    "episodes": 1000,
    "avg_reward": -85.23,
    "std_reward": 5.67,
    "final_reward": -82.10,
    "best_reward": -75.45,
    "avg_length": 3600,
    "eval_rewards": [-85.0, -83.5, -82.0, ...],
    "final_eval_reward": -80.5,
    "training_losses": [0.123, 0.098, ...],
    "training_time": 1234.56,
    "episode_rewards": [-100.0, -98.5, ...],
    "episode_lengths": [3600, 3600, ...]
  },
  "SAC": { ... },
  "Rainbow DQN": { ... }
}
```

## Recommended Training Configurations

### Quick Test (Development)
```bash
python scripts/train_phase6_full_scale.py --episodes 100 --eval-interval 20
```
**Time**: ~10-15 minutes per algorithm

### Standard Training
```bash
python scripts/train_phase6_full_scale.py --episodes 1000 --eval-interval 100
```
**Time**: ~2-3 hours per algorithm

### Full-Scale Training
```bash
python scripts/train_phase6_full_scale.py --episodes 2000 --eval-interval 100
```
**Time**: ~4-6 hours per algorithm

### Production Training
```bash
python scripts/train_phase6_full_scale.py --episodes 5000 --eval-interval 200 --save-interval 500
```
**Time**: ~10-15 hours per algorithm

## Monitoring Training

### Real-time Logs
Training logs are written to:
- Console output (stdout)
- `phase6_training.log` file

### Key Metrics to Monitor

1. **Average Reward**: Should increase over time
2. **Final Eval Reward**: Performance on evaluation episodes
3. **Training Loss**: Should decrease (for PPO, SAC)
4. **Episode Length**: Should stabilize

### Example Output
```
Episode 100/1000 | Avg Reward: -95.23 | Eval Reward: -92.10
Episode 200/1000 | Avg Reward: -88.45 | Eval Reward: -85.30
Episode 300/1000 | Avg Reward: -82.10 | Eval Reward: -78.50
...
```

## Hyperparameter Optimization Workflow

### Step 1: Optimize Hyperparameters
```bash
# Optimize each algorithm
python scripts/optimize_phase6.py --algorithm PPO --trials 50
python scripts/optimize_phase6.py --algorithm SAC --trials 50
python scripts/optimize_phase6.py --algorithm "Rainbow DQN" --trials 50
```

### Step 2: Review Optimization Results
Check `./runs/phase6_optimization/` for:
- Best parameters JSON files
- Optimization history plots
- Parameter importance plots

### Step 3: Train with Optimized Parameters
```bash
python scripts/train_phase6_full_scale.py \
    --episodes 1000 \
    --use-optimized \
    --optimized-params ./runs/phase6_optimization/ppo_best_params.json
```

## Performance Expectations

Based on OPTIMIZATION_ROADMAP.md Phase 6 specifications:

| Algorithm | Expected Improvement | Training Time (1000 episodes) |
|-----------|---------------------|------------------------------|
| PPO | 20-25% sample efficiency | ~2-3 hours |
| SAC | 15-20% performance, 25% sample efficiency | ~2-3 hours |
| Rainbow DQN | 30-35% performance | ~3-4 hours |

### Baseline Comparison
- **Baseline DQN**: ~-107.81 avg reward
- **Target (Phase 6)**: ~-70 to -75 avg reward (50-55% improvement)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `buffer_size` in config
- Train one algorithm at a time

### Slow Training
- Reduce `episodes` for testing
- Use GPU if available (automatic)
- Reduce `eval_interval` to check progress less frequently

### Poor Performance
- Check hyperparameters
- Run hyperparameter optimization
- Increase training episodes
- Check environment configuration

## Advanced Usage

### Custom Configuration
Create a custom config file and use it:

```bash
python scripts/train_phase6_full_scale.py \
    --config configs/custom_intersection.json \
    --episodes 1000
```

### Resume Training
Currently, training starts from scratch. To resume:
1. Load saved checkpoint models
2. Continue training from checkpoint episode

### Distributed Training
For large-scale experiments, run multiple instances:

```bash
# Terminal 1
python scripts/train_phase6_full_scale.py --algorithm PPO --episodes 1000

# Terminal 2
python scripts/train_phase6_full_scale.py --algorithm SAC --episodes 1000

# Terminal 3
python scripts/train_phase6_full_scale.py --algorithm "Rainbow DQN" --episodes 1000
```

## Results Analysis

### Compare Algorithms
```python
import json

with open('runs/phase6_full_scale/training_results.json') as f:
    results = json.load(f)

for alg, data in results.items():
    print(f"{alg}:")
    print(f"  Avg Reward: {data['avg_reward']:.2f}")
    print(f"  Final Eval: {data['final_eval_reward']:.2f}")
```

### Plot Training Curves
Use the episode rewards from `training_results.json` to plot:
- Learning curves
- Reward distributions
- Convergence analysis

## Next Steps

After training:
1. **Analyze Results**: Review `training_summary.txt`
2. **Compare Performance**: Compare against baseline algorithms
3. **Select Best Algorithm**: Choose best performer for deployment
4. **Fine-tune**: Further optimize best algorithm
5. **Deploy**: Use best model for production

---

**For questions or issues, check:**
- `PHASE_6_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `PHASE_6_TEST_VALIDATION_REPORT.md` - Test results
- `OPTIMIZATION_ROADMAP.md` - Phase 6 specifications

