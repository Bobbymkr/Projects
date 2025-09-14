# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This repository implements an **Adaptive Traffic Signal Control System** using Deep Q-Network (DQN) reinforcement learning. It supports single-agent control, multi-agent reinforcement learning (MARL), SUMO simulation integration, and real-time video-based traffic analysis.

## Quick Start

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For SUMO support, install eclipse-sumo
pip install eclipse-sumo

# For TensorFlow (required for traffic forecasting in MARL mode)
pip install tensorflow

# Set up environment variables
# Windows PowerShell:
$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"
# macOS/Linux:
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### SUMO Setup (Optional)
If using SUMO-based environments:
```bash
# Install SUMO (if not using pip version)
# Windows: Download from https://sumo.dlr.de/docs/Downloads.php
# macOS: brew install sumo
# Linux: sudo apt-get install sumo sumo-tools

# Set SUMO_HOME environment variable
# The pip install eclipse-sumo handles this automatically
```

## Common Commands

### Training

#### 1. Basic Training (TrafficEnv)
Train a DQN agent on the synthetic traffic environment:
```bash
python src\rl\train_dqn.py --episodes 5 --config configs/intersection.json --out runs
```

#### 2. SUMO-based Training
Train using SUMO simulation:
```bash
python src\rl\train_dqn.py --use_sumo --episodes 5 --config configs/intersection.json --out runs
```

#### 3. Multi-Agent RL Training (MARL)
Train multiple agents for grid network coordination:
```bash
python src\rl\train_dqn.py --marl --config configs/grid.sumocfg --episodes 5 --out runs
```

#### 4. Resume Training from Checkpoint
Continue training from a saved checkpoint:
```bash
python src\rl\train_dqn.py --episodes 10 --config configs/intersection.json --out runs
# (Automatically resumes from runs/checkpoint if it exists)
```

### Inference

#### 1. Simulation-based Inference
Run inference on simulated environment:
```bash
python src\rl\inference.py sim --model runs/dqn_traffic.npz --config configs/intersection.json
```

#### 2. Video-based Inference
Process video file with trained model:
```bash
python src\rl\inference.py video --model runs/dqn_traffic.npz --video_source path/to/video.mp4
```

#### 3. Real-time Camera Inference
Use webcam for real-time inference:
```bash
python src\rl\inference.py video --model runs/dqn_traffic.npz --video_source 0
```

#### 4. MARL Inference
Run multi-agent inference:
```bash
python src\rl\inference.py sim --model runs --marl --config configs/grid.sumocfg
```

### Visualization and Analysis

#### 1. Visualize Training Progress
Plot queue dynamics over time:
```bash
python src\rl\visualize_sim.py --config configs/intersection.json --model runs/dqn_traffic.npz --steps 30 --outdir runs
```

#### 2. Compare Strategies
Compare DQN vs fixed-timing strategies across different traffic scenarios:
```bash
python src\rl\compare_strategies.py
```
This automatically tests on morning rush, normal, and evening rush configurations.

#### 3. Analyze DQN Performance
Detailed analysis with multiple metrics:
```bash
python src\rl\analyze_dqn.py
```

#### 4. Simple DQN Analysis
Quick performance check:
```bash
python src\rl\analyze_dqn_simple.py
```

### Live Simulations

#### 1. Live Simulation with Visualization
Run interactive simulation with real-time visualization:
```bash
python src\rl\live_simulation.py
```

#### 2. Inference with Visualization
Run inference with visual output:
```bash
python src\rl\inference_viz.py
```

#### 3. SUMO GUI Mode
When using SUMO, add GUI visualization during training/inference by modifying the SUMO binary in the code from 'sumo' to 'sumo-gui'.

## Architecture Overview

### Core Components

1. **Environments** (`src/env/`)
   - `TrafficEnv`: Synthetic traffic environment with configurable parameters
   - `SumoEnv`: SUMO-based environment using TraCI interface
   - `MarlEnv`: Multi-agent environment for network-wide coordination
   - `VideoTrafficEnv`: Real-world video input processing environment

2. **RL Agent** (`src/rl/`)
   - `DQNAgent`: Deep Q-Network implementation with:
     - Experience replay buffer (size: 10,000)
     - Target network (update frequency: 100 steps)
     - ε-greedy exploration (ε: 1.0 → 0.01)
     - Adam optimizer (learning rate: 0.001)
   - Neural Network: 3-layer MLP (input → 128 → 128 → output)

3. **Vision Pipeline** (`src/vision/`)
   - `YOLOQueueEstimator`: YOLOv8-based vehicle detection and queue estimation
   - `VideoInputStream`: Multi-source video input handling (webcam, file, RTSP)
   - `ROIManager`: Region of Interest management for lane-specific detection
   - Automatic model download if `yolov8n.pt` not present

4. **Traffic Forecasting** (`src/forecast/`)
   - `TrafficForecaster`: LSTM-based traffic prediction for MARL
   - Predicts future traffic states for enhanced decision-making
   - Input: Historical traffic data (10 timesteps)
   - Output: Future predictions (5 timesteps)

### State and Action Spaces

#### TrafficEnv / SumoEnv
- **State**: Queue lengths for each lane (4-dimensional vector)
- **Actions**: Green light durations (5-60 seconds, 5-second steps = 12 actions)
- **Reward**: `-(queue_weight × total_queue) - (wait_weight × total_wait)`

#### MarlEnv
- **State**: [Own queues (8)] + [Neighbor queues (24)] + [Predictions (40)] = 72-dimensional
- **Actions**: Binary (change phase or maintain) per intersection
- **Reward**: Queue penalty + Wait penalty + Phase flicker penalty

### Configuration Files

1. **Intersection Configs** (`configs/`)
   - `intersection.json`: Default single intersection setup
   - `morning_rush.json`: High traffic morning scenario
   - `evening_rush.json`: High traffic evening scenario

2. **SUMO Configs** (`configs/`)
   - `sumo.sumocfg`: Single intersection SUMO config
   - `grid.sumocfg`: 2x2 grid network for MARL
   - Network files: `.nod.xml`, `.edg.xml`, `.net.xml`
   - Route files: `.rou.xml`, `.trips.xml`

### Key Parameters

```json
{
  "num_lanes": 4,
  "phase_lanes": [[0,1], [2,3]],
  "min_green": 5,
  "max_green": 60,
  "green_step": 5,
  "cycle_yellow": 3,
  "cycle_all_red": 1,
  "arrival_rates": [0.3, 0.25, 0.35, 0.2],
  "queue_capacity": 40,
  "reward_weights": {
    "queue": -1.0,
    "wait_penalty": -0.1
  }
}
```

## Development Tips

### Running Tests
Currently, there are no automated tests. Validate changes by:
1. Running a short training episode: `python src\rl\train_dqn.py --episodes 1`
2. Running inference on the trained model
3. Checking visualization outputs

### Performance Optimization
- Reduce video resolution for faster processing: Modify `frame_width` and `frame_height` in video config
- Disable visualization during training for better performance
- Use smaller replay buffer size for memory-constrained systems

### Debugging
- Enable SUMO GUI by changing 'sumo' to 'sumo-gui' in `sumo_env.py`
- Check queue visualization: `python src\rl\visualize_sim.py`
- Monitor training progress via saved rewards in `runs/rewards.npy`

## Troubleshooting

### Common Issues

1. **SUMO_HOME not set**
   - Solution: Install via `pip install eclipse-sumo` or set manually after installation

2. **Module import errors**
   - Solution: Ensure `PYTHONPATH` includes the project root
   - Windows: `$env:PYTHONPATH = "$pwd;$env:PYTHONPATH"`
   - Unix: `export PYTHONPATH=$(pwd):$PYTHONPATH`

3. **Video source not found**
   - For webcam: Use index (0, 1, etc.)
   - For files: Use absolute paths
   - For RTSP: Ensure network connectivity

4. **GPU/CUDA issues**
   - The codebase uses CPU by default
   - PyTorch operations will use CUDA if available
   - No explicit GPU configuration needed

5. **Training convergence issues**
   - Try different reward weights in configuration
   - Adjust learning rate in `DQNConfig`
   - Increase replay buffer size for more stable learning

### Model Checkpoints
- Training automatically saves checkpoints to `runs/checkpoint/`
- Final models saved to `runs/dqn_traffic.npz` (or with agent suffix for MARL)
- Checkpoints include both model weights and training rewards

## Additional Resources
- SUMO Documentation: https://sumo.dlr.de/docs/
- Gymnasium Documentation: https://gymnasium.farama.org/
- YOLOv8 Documentation: https://docs.ultralytics.com/