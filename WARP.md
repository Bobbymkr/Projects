# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Commands

### Environment Setup (Windows PowerShell)
```powershell
# Allow script execution for this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # For development with testing tools
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m unit          # Unit tests
pytest tests/ -m integration   # Integration tests
pytest tests/ -m system        # System tests
pytest tests/ -m perf          # Performance tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_traffic_env.py -v
```

### Training DQN Models
```bash
# Quick training (5 episodes)
python src/rl/train_dqn.py --episodes 5 --config configs/intersection.json

# Production training (6000 episodes)
python src/rl/train_dqn_pytorch.py --episodes 6000 --out runs/production

# Multi-agent training
python src/rl/train_dqn.py --episodes 100 --marl --config configs/grid.sumocfg

# With hyperparameter tuning
python src/rl/train_dqn.py --episodes 100 --tune --config configs/intersection.json

# With SUMO simulation
python src/rl/train_dqn.py --episodes 100 --use_sumo --config configs/grid.sumocfg
```

### Running Demos
```bash
# Basic demo
python demo.py

# Working demo (simplified, no dependencies)
python working_demo.py

# Professional demo with all features
python demo_professional.py

# Forecast demo
python demo_forecast.py
```

### Inference
```bash
# Use webcam for real-time control
python src/rl/inference.py video --model runs/dqn_traffic.npz --video_source 0

# Process video file
python src/rl/inference.py video --model runs/dqn_traffic.npz --video_source traffic.mp4

# Simulation inference
python src/rl/inference.py sim --model runs/dqn_traffic.npz --episodes 10
```

### Benchmarking
```bash
# Benchmark different algorithms
python src/rl/benchmark_methods.py

# Evaluate trained agent
python evaluate_700ep_agent.py

# Quality assessment
python comprehensive_accuracy_assessment.py
```

### Code Quality
```bash
# Format code with black
black src/ tests/ --line-length=88

# Run linting with ruff
ruff check src/
ruff check src/ --fix  # Auto-fix issues

# Type checking with mypy
mypy src/
```

### Build & Development
```bash
# Using Makefile (Linux/Mac/WSL)
make install         # Install package
make install-dev     # Install with dev dependencies
make test           # Run tests
make test-cov       # Run tests with coverage
make format         # Format code
make lint           # Run linting
make run-demo       # Run demo
make clean          # Clean build artifacts
```

## High-Level Architecture

### Core Components

**Perception Layer:**
- **YOLOv8 Detection** (`src/vision/`): Vehicle detection and tracking using ultralytics YOLOv8
- **Queue Estimation**: Analyzes detection outputs to estimate queue lengths per lane

**Environment Layer:**
- **TrafficEnv** (`src/env/traffic_env.py`): Simulated traffic environment with configurable intersections
- **SUMOEnv** (`src/env/sumo_env.py`): Integration with SUMO traffic simulator for realistic simulations
- **VideoEnv** (`src/env/video_env.py`): Real-time video processing environment for camera inputs
- **MARLEnv** (`src/env/marl_env.py`): Multi-agent environment for city-wide coordination

**Decision Layer:**
- **DQN Agent** (`src/rl/`): Deep Q-Network reinforcement learning agents (standard, double, dueling)
- **Fuzzy Controller** (`src/control/fuzzy_controller.py`): Rule-based fuzzy logic control
- **Webster's Method** (`src/control/webster.py`): Traditional fixed-time signal optimization
- **Genetic Algorithm** (`src/control/ga_controller.py`): Evolutionary optimization approach
- **PSO Controller** (`src/control/pso_controller.py`): Particle Swarm Optimization
- **GNN Controller** (`src/control/`): Graph Neural Network for network-level optimization

**Forecasting:**
- **CNN-LSTM Models** (`src/forecast/`): Traffic demand prediction using deep learning

### Data Flow
1. **Input**: Video stream/camera → YOLOv8 detection
2. **Processing**: Detection outputs → State encoder → Queue/density metrics
3. **Decision**: State → RL Agent/Controller → Action selection
4. **Control**: Action → Signal phase/timing adjustment
5. **Feedback**: Environment response → Reward calculation → Agent learning

## Configuration Structure

### Basic Intersection Configuration (JSON)
```json
{
  "num_lanes": 4,
  "phase_lanes": [[0, 1], [2, 3]],
  "min_green": 5,
  "max_green": 60,
  "green_step": 5,
  "arrival_rates": [0.3, 0.25, 0.35, 0.2],
  "queue_capacity": 40,
  "reward_weights": {
    "queue": -1.0,
    "wait_penalty": -0.1
  }
}
```

### Available Scenarios
Located in `configs/` directory:
- **balanced**: Equal traffic from all directions (`configs/intersection.json`)
- **morning_rush**: Heavy eastbound traffic (`configs/morning_rush.json`)
- **evening_rush**: Heavy westbound traffic (`configs/evening_rush.json`)
- **north_heavy**: Dominant north-south flow (`configs/north_heavy.json`)
- **cross_flow**: Diagonal traffic patterns (`configs/cross_flow.json`)

## Performance Metrics

### Key Metrics Tracked
- **Wait Time**: Average vehicle wait time at intersection
- **Queue Length**: Number of vehicles waiting per lane
- **Throughput**: Vehicles processed per hour
- **Efficiency**: Ratio of green time utilization
- **Travel Time**: Total journey time through network
- **Emissions**: Estimated CO2 based on idle/acceleration

### Performance Results (from README)
| Algorithm | Wait Time | Queue Length | Efficiency | Grade |
|-----------|-----------|--------------|------------|-------|
| Fuzzy Control | 8.51s | 12.5 vehicles | 1.2123 | A+ |
| GNN Forecasting | 13.58s | 15.4 vehicles | 1.1848 | A |
| DQN (6000 episodes) | 21.47s | 23.4 vehicles | 1.2064 | B+ |
| Traditional (Webster) | 27.37s | 24.9 vehicles | 1.1612 | C |

## Test Markers

The project uses pytest markers to categorize tests:
- **unit**: Fast, isolated unit tests
- **integration**: Tests requiring component interactions
- **system**: End-to-end system tests
- **perf**: Performance/benchmark tests
- **vision**: Computer vision pipeline tests
- **forecasting**: Traffic forecasting tests
- **sumo**: Tests requiring SUMO simulation
- **gpu**: Tests requiring GPU acceleration
- **slow**: Tests taking > 30 seconds
- **nightly**: Tests for nightly builds only
- **flaky**: Known unreliable tests

## Codacy Integration Rules

Based on `.github/instructions/codacy.instructions.md`:

### CRITICAL: After ANY file edits
- **MUST run** `codacy_cli_analyze` tool for each edited file
- Set `rootPath` to workspace path and `file` to edited file path
- If issues are found, propose and apply fixes immediately
- Failure to follow this is considered a critical error

### After dependency changes
When adding packages to requirements.txt, package.json, pom.xml, etc.:
- **MUST run** `codacy_cli_analyze` with `tool: "trivy"` for security scanning
- Stop all operations if vulnerabilities are found
- Fix security issues before continuing

### General Rules
- Do not manually install Codacy CLI with brew/npm/npx
- Use the Codacy MCP Server tools when available
- For 404 errors, offer to run `codacy_setup_repository`

## Important Notes

### Windows-Specific Setup
- Use PowerShell with execution policy bypass for virtual environment activation
- SUMO installation requires setting `SUMO_HOME` environment variable
- For camera stability: set `OPENCV_VIDEOIO_PRIORITY_MSMF=0`

### GPU Support
- For CUDA GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- For CPU only: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Development Workflow
1. Make code changes
2. Run `black` for formatting
3. Run `ruff check` for linting
4. Run tests with `pytest`
5. Run `codacy_cli_analyze` before committing
6. Commit with conventional commit messages

## Maintenance

When updating this repository:
- Update commands section when new scripts/configs are added
- Update architecture section when new algorithms/controllers are introduced
- Update test markers when pytest configuration changes
- Keep performance metrics current with latest benchmarks