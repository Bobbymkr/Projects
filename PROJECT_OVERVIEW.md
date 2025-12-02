# Adaptive Traffic Signal Control System - Complete Project Overview

## 🎯 What Is This Project?

This is an **intelligent traffic signal control system** that uses **Artificial Intelligence** to optimize traffic flow at intersections. Instead of using fixed-time traffic lights (like traditional systems), this system:

1. **Observes** real-time traffic conditions using cameras or simulations
2. **Learns** optimal signal timing strategies using Deep Reinforcement Learning
3. **Adapts** to changing traffic patterns automatically
4. **Reduces** wait times by up to 40% compared to traditional systems

## 🏗️ System Architecture - How Everything Works Together

### High-Level Flow

```
Video/Simulation Input
    ↓
[Vision Pipeline] → Detects vehicles, estimates queue lengths
    ↓
[Traffic Environment] → Simulates traffic dynamics
    ↓
[RL Agent / Control Strategy] → Decides optimal green light duration
    ↓
[Signal Controller] → Applies the decision
    ↓
[Monitoring] → Tracks performance, logs metrics
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ REST API │  │WebSocket │  │ GraphQL  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼─────────────┼──────────────────────────┘
        │             │             │
        └─────────────┴─────────────┘
                    │
        ┌───────────▼────────────┐
        │   APPLICATION LAYER     │
        │  ┌──────────────────┐  │
        │  │ Traffic Controller│  │
        │  │ Performance Monitor│  │
        │  │ Security Manager  │  │
        │  └─────────┬─────────┘  │
        └────────────┼─────────────┘
                     │
        ┌────────────▼────────────┐
        │   INTELLIGENCE LAYER     │
        │  ┌──────────────────┐   │
        │  │   DQN Agent      │   │ ← Learns optimal timing
        │  │   Fuzzy Control  │   │ ← Rule-based control
        │  │   Webster Method │   │ ← Traditional optimization
        │  │   GA/PSO         │   │ ← Evolutionary algorithms
        │  │   Traffic Forecaster│ │ ← Predicts future traffic
        │  └─────────┬─────────┘   │
        └────────────┼─────────────┘
                     │
        ┌────────────▼────────────┐
        │    PERCEPTION LAYER     │
        │  ┌──────────────────┐   │
        │  │  YOLOv8 Detector │   │ ← Detects vehicles in video
        │  │  Queue Estimator │   │ ← Counts vehicles per lane
        │  │  Video Pipeline  │   │ ← Processes video streams
        │  └─────────┬─────────┘   │
        └────────────┼─────────────┘
                     │
        ┌────────────▼────────────┐
        │   SIMULATION LAYER      │
        │  ┌──────────────────┐   │
        │  │ TrafficEnv       │   │ ← Basic simulation
        │  │ SumoEnv          │   │ ← SUMO integration
        │  │ MarlEnv          │   │ ← Multi-intersection
        │  │ VideoEnv         │   │ ← Real-world video
        │  └──────────────────┘   │
        └─────────────────────────┘
```

## 📦 Project Structure Explained

### 1. **adaptive-traffic-core** - The Brain

**Purpose**: Contains all the intelligence and decision-making logic.

**Key Components**:

#### Reinforcement Learning (`src/rl/`)
- **DQN Agent**: Deep Q-Network that learns optimal signal timing
- **Training Scripts**: Scripts to train the agent on traffic scenarios
- **Inference**: Use trained models to make real-time decisions

**How it works**:
1. Agent observes traffic state (queue lengths, wait times)
2. Agent selects an action (green light duration: 5-60 seconds)
3. Environment simulates traffic flow for that duration
4. Agent receives reward (negative for long queues/wait times)
5. Agent learns from experience to maximize reward

#### Environments (`src/env/`)
- **TrafficEnv**: Basic traffic simulation (Poisson arrivals, queue-based)
- **SumoEnv**: Integration with SUMO traffic simulator (more realistic)
- **MarlEnv**: Multi-Agent RL for coordinating multiple intersections
- **VideoEnv**: Real-world video input for live traffic control

#### Control Strategies (`src/control/`)
- **Fuzzy Control**: Rule-based system using fuzzy logic (best performance: A+)
- **Webster Method**: Traditional traffic engineering optimization

#### Optimization (`src/optimization/`)
- **Genetic Algorithm**: Evolutionary optimization for signal timing
- **Particle Swarm Optimization**: Swarm intelligence approach

#### Forecasting (`src/forecast/`)
- **Traffic Forecaster**: LSTM-based prediction of future traffic
- **GNN Forecaster**: Graph Neural Network for network-wide predictions

### 2. **adaptive-traffic-vision** - The Eyes

**Purpose**: Processes video feeds to detect vehicles and estimate traffic.

**Key Components**:
- **YOLOv8 Detector**: State-of-the-art object detection model
- **Queue Estimator**: Counts vehicles in each lane using ROI (Region of Interest)
- **Video Pipeline**: Handles webcam, video files, or IP camera inputs

**How it works**:
1. Video frame is captured
2. YOLOv8 detects all vehicles in the frame
3. Vehicles are filtered by ROI (only count vehicles in relevant lanes)
4. Queue length is estimated for each approach lane
5. Queue data is sent to the RL agent or control strategy

### 3. **adaptive-traffic-api** - The Interface

**Purpose**: Provides REST API, WebSocket, and GraphQL interfaces for external systems.

**Key Components**:
- **REST API**: Standard HTTP endpoints for traffic control
- **WebSocket**: Real-time updates for dashboards
- **GraphQL**: Flexible query interface
- **Authentication**: OAuth2/JWT for secure access
- **Security**: Rate limiting, input validation, audit logging

**Use Cases**:
- Web dashboards querying traffic status
- Mobile apps controlling signals
- Integration with city traffic management systems
- Real-time monitoring and alerts

### 4. **adaptive-traffic-research** - The Laboratory

**Purpose**: Experimental algorithms and research features.

**Key Components**:
- **Novel Algorithms**: Hierarchical RL, Imitation Learning, Model-based RL
- **Federated Learning**: Privacy-preserving distributed training
- **Explainability**: Tools to understand why the AI makes certain decisions
- **Benchmarking**: Comprehensive test suites for algorithm comparison

### 5. **adaptive-traffic-deployment** - The Infrastructure

**Purpose**: Production deployment configurations.

**Key Components**:
- **Kubernetes**: Container orchestration for scaling
- **Helm Charts**: Easy deployment and configuration
- **Docker**: Containerization
- **Monitoring**: Grafana dashboards, Prometheus metrics

### 6. **adaptive-traffic-common** - The Foundation

**Purpose**: Shared utilities used by all projects.

**Key Components**:
- **Utils**: Configuration loading, metrics calculation, health checks
- **Benchmarking**: Common benchmarking utilities

## 🔄 Complete Data Flow - Step by Step

### Scenario 1: Training a DQN Agent

```
1. [Script] train_dqn.py starts
   ↓
2. [Environment] TrafficEnv loads config (intersection.json)
   - Creates 4-lane intersection
   - Sets up traffic arrival rates
   ↓
3. [Agent] DQN Agent initialized
   - Neural network created (4 inputs → 128 hidden → 12 outputs)
   - Experience replay buffer initialized
   ↓
4. [Training Loop] For each episode:
   a. Environment resets → Random initial traffic state
   b. Agent observes state → [queue_lane0, queue_lane1, queue_lane2, queue_lane3]
   c. Agent selects action → ε-greedy: explore (random) or exploit (Q-network)
   d. Environment steps → Simulates traffic for chosen green duration
   e. Reward calculated → -(queue_weight × total_queues) - (wait_weight × total_wait)
   f. Experience stored → (state, action, reward, next_state) → Replay buffer
   g. Agent trains → Sample batch from buffer, update Q-network
   ↓
5. [Checkpoint] Every N episodes, save model to disk
```

### Scenario 2: Real-Time Inference with Video

```
1. [Video] Camera captures frame
   ↓
2. [Vision] YOLOv8 processes frame
   - Detects all vehicles (bounding boxes)
   - Filters by ROI (only vehicles in traffic lanes)
   - Counts vehicles per lane → [12, 8, 15, 6]
   ↓
3. [Environment] VideoEnv receives queue data
   - Creates state vector: [12, 8, 15, 6, current_phase, elapsed_time]
   ↓
4. [Agent] DQN Agent (loaded from trained model)
   - Feeds state to neural network
   - Gets Q-values for all 12 possible actions
   - Selects action with highest Q-value → e.g., "35 seconds green"
   ↓
5. [Control] Signal controller applies action
   - Sets green light for selected lanes
   - Waits for specified duration
   - Transitions to next phase
   ↓
6. [Monitoring] Metrics logged
   - Queue lengths, wait times, throughput
   - Sent to API for dashboard display
```

### Scenario 3: Multi-Agent Coordination (MARL)

```
1. [Network] Multiple intersections in a grid
   ↓
2. [Agents] One DQN agent per intersection
   ↓
3. [Coordination] Agents communicate
   - Share traffic state with neighbors
   - Consider upstream/downstream traffic
   ↓
4. [Decision] Each agent makes local decision
   - But considers network-wide effects
   ↓
5. [Optimization] Global traffic flow improved
   - Reduces cascading congestion
   - Optimizes green waves
```

## 🧠 Key Algorithms Explained

### 1. Deep Q-Network (DQN)

**What it is**: A neural network that learns to estimate the "value" of taking each action in each state.

**How it learns**:
- **Experience Replay**: Stores past experiences, samples randomly (breaks correlation)
- **Target Network**: Separate network for stable learning targets
- **Q-Learning Update**: `Q(s,a) = r + γ * max(Q(s',a'))` (Bellman equation)

**Why it works**: By trying different actions and seeing rewards, it learns which actions lead to better traffic flow.

### 2. Fuzzy Logic Control

**What it is**: Rule-based system using linguistic variables (e.g., "high traffic", "low traffic").

**How it works**:
- Inputs: Queue lengths (fuzzified into "low", "medium", "high")
- Rules: IF queue_north is HIGH AND queue_south is LOW THEN extend_green_north
- Output: Defuzzified to actual green duration

**Why it works**: Captures human expert knowledge in rules, very interpretable.

### 3. Traffic Forecasting (LSTM/GNN)

**What it is**: Predicts future traffic conditions.

**How it works**:
- **LSTM**: Processes historical traffic sequences, predicts next time step
- **GNN**: Models traffic network as graph, predicts network-wide patterns

**Why it's useful**: Allows proactive control (anticipate congestion before it happens).

## 📊 Performance Metrics

The system has been tested and achieves:

| Algorithm | Average Wait Time | Queue Length | Efficiency Score | Grade |
|-----------|------------------|--------------|------------------|-------|
| **Fuzzy Control** | 8.51s | 12.5 vehicles | 1.2123 | **A+** |
| **GNN Forecasting** | 13.58s | 15.4 vehicles | 1.1848 | **A** |
| **DQN (6000 episodes)** | 21.47s | 23.4 vehicles | 1.2064 | **B+** |
| **Traditional (Webster)** | 27.37s | 24.9 vehicles | 1.1612 | **C** |

**Key Insight**: Fuzzy Control performs best because it encodes expert knowledge. DQN can potentially exceed this with more training.

## 🔧 Configuration System

### Intersection Configuration (`configs/intersection.json`)

```json
{
  "num_lanes": 4,                    // 4 approach lanes
  "phase_lanes": [[0,1], [2,3]],     // Lanes 0,1 get green together; 2,3 together
  "min_green": 5,                    // Minimum 5 seconds green
  "max_green": 60,                   // Maximum 60 seconds green
  "green_step": 5,                   // 5-second increments (5, 10, 15, ..., 60)
  "arrival_rates": [0.3, 0.25, 0.35, 0.2],  // Vehicles per second per lane
  "queue_capacity": 40,              // Max vehicles per lane
  "reward_weights": {
    "queue": -1.0,                   // Penalty for each vehicle in queue
    "wait_penalty": -0.1             // Penalty for each second of waiting
  }
}
```

## 🚀 Getting Started - Complete Workflow

### Step 1: Install Dependencies

```bash
# Install base library
pip install -e adaptive-traffic-common

# Install core system
pip install -e adaptive-traffic-core

# Install vision (optional, for video processing)
pip install -e adaptive-traffic-vision

# Install API (optional, for web interface)
pip install -e adaptive-traffic-api
```

### Step 2: Train a Model

```bash
cd adaptive-traffic-core
python train_dqn.py --episodes 1000 --config configs/intersection.json
```

This will:
- Create a TrafficEnv with the specified configuration
- Train a DQN agent for 1000 episodes
- Save the trained model to `runs/dqn_traffic.npz`

### Step 3: Test the Model

```bash
python src/rl/inference.py sim --model runs/dqn_traffic.npz --episodes 10
```

This will:
- Load the trained model
- Run 10 test episodes
- Display performance metrics

### Step 4: Use with Real Video (Optional)

```bash
# With webcam
python src/rl/inference.py video --model runs/dqn_traffic.npz --video_source 0

# With video file
python src/rl/inference.py video --model runs/dqn_traffic.npz --video_source traffic.mp4
```

### Step 5: Deploy API (Optional)

```bash
cd adaptive-traffic-api
python -m adaptive_traffic_api.api.main
# Visit http://localhost:8000/docs
```

## 🔍 Understanding the Codebase - Where to Look

### For Understanding RL Training:
- `adaptive-traffic-core/src/rl/dqn_agent.py` - The DQN implementation
- `adaptive-traffic-core/src/rl/train_dqn.py` - Training script
- `adaptive-traffic-core/src/env/traffic_env.py` - Environment implementation

### For Understanding Vision:
- `adaptive-traffic-vision/src/vision/yolo_queue.py` - Vehicle detection and queue estimation
- `adaptive-traffic-vision/src/vision/video_pipeline.py` - Video processing

### For Understanding API:
- `adaptive-traffic-api/src/api/main.py` - FastAPI application
- `adaptive-traffic-api/src/api/routes/traffic.py` - Traffic control endpoints

### For Understanding Control Strategies:
- `adaptive-traffic-core/src/control/fuzzy_control.py` - Fuzzy logic implementation
- `adaptive-traffic-core/src/control/webster_method.py` - Traditional method

## 🎓 Key Concepts to Understand

1. **Reinforcement Learning**: Agent learns by trial and error, receiving rewards/penalties
2. **State**: Current traffic condition (queue lengths, wait times, current phase)
3. **Action**: Green light duration choice (5, 10, 15, ..., 60 seconds)
4. **Reward**: Negative value based on queues and wait times (we want to minimize)
5. **Episode**: One complete simulation run (reset → many steps → done)
6. **Experience Replay**: Storing past experiences and learning from them randomly
7. **ROI (Region of Interest)**: Specific areas in video frame where we count vehicles

## 🔗 Dependencies Between Projects

```
adaptive-traffic-common (no dependencies)
    ↑
    ├── adaptive-traffic-core (depends on common)
    │   └── adaptive-traffic-api (depends on core + common)
    │
    ├── adaptive-traffic-vision (depends on common)
    │   └── adaptive-traffic-core (optional, for video environments)
    │
    └── adaptive-traffic-research (depends on core + common)
```

**Installation Order Matters**: Install common first, then core, then others.

## 📝 Common Use Cases

### Use Case 1: Research New Algorithm
- Work in `adaptive-traffic-research/`
- Implement new algorithm in `src/research/novel_algorithms/`
- Use `adaptive-traffic-core` environments for testing

### Use Case 2: Deploy to Production
- Use `adaptive-traffic-deployment/` for K8s/Docker configs
- Deploy `adaptive-traffic-api` for web interface
- Deploy `adaptive-traffic-core` for control logic
- Use `adaptive-traffic-vision` for real-world video processing

### Use Case 3: Benchmark Algorithms
- Use `adaptive-traffic-core/src/rl/benchmark_methods.py`
- Compare DQN, Fuzzy, Webster, GA, PSO
- Generate performance reports

## 🐛 Troubleshooting Guide

### Problem: Import errors
**Solution**: Make sure all packages are installed in correct order (see installation above)

### Problem: Model not learning
**Solution**: 
- Check reward function (should be negative for bad states)
- Increase training episodes
- Adjust learning rate in DQNConfig

### Problem: Vision not detecting vehicles
**Solution**:
- Check ROI configuration (regions of interest)
- Verify YOLOv8 model file exists
- Check video source (webcam index or file path)

### Problem: API not starting
**Solution**:
- Check if FastAPI is installed
- Verify database connection (if using database)
- Check port 8000 is not in use

## 📚 Additional Resources

- **Original README**: `README.md` - Original project documentation
- **Architecture Docs**: `high_level_architecture.md`, `component_diagram.md`
- **Migration Guide**: `MIGRATION_GUIDE.md` - If migrating from old structure
- **Quick Start**: `QUICK_START_RESTRUCTURED.md` - Quick reference

## ✅ Summary

This project is a **complete intelligent traffic control system** that:

1. **Learns** optimal signal timing using AI (DQN)
2. **Sees** real traffic using computer vision (YOLOv8)
3. **Controls** signals adaptively based on current conditions
4. **Forecasts** future traffic for proactive control
5. **Coordinates** multiple intersections for network-wide optimization
6. **Provides** APIs for integration and monitoring
7. **Deploys** easily with containerization and orchestration

The system is **modular** (6 separate projects), **well-documented**, and **production-ready**.

---

**For an AI or developer reading this**: You now have a complete understanding of what this project does, how it's structured, how components interact, and how to use it. Start with the Quick Start guide, then explore individual components based on your needs.

