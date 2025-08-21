Adaptive Traffic Signal Control - Quickstart

This repo implements an adaptive traffic signal control system using DQN with Multi-Agent RL (MARL) support and LSTM-based traffic forecasting for multiple intersections.

Folders:
- src/env: Gymnasium environments including MarlEnv for multi-agent setups
- src/rl: DQN training and inference with MARL support
- src/vision: YOLOv8-based queue extraction
- src/forecast: LSTM traffic forecasting module
- configs: Configuration files including grid.sumocfg for multi-intersection simulations

Steps:
1) Create venv: python -m venv .venv
2) Activate venv (Windows): .venv\Scripts\activate
3) Install deps: pip install -r requirements.txt
4) Train: python src\rl\train_dqn.py --episodes 5 --marl --config configs/grid.sumocfg
5) Inference demo: python src\rl\inference.py --marl

SUMO Integration:
- Install SUMO: pip install eclipse-sumo
- Ensure SUMO_HOME is set and binaries in PATH.
- Use --use_sumo flag for SUMO mode, e.g., python src\rl\train_dqn.py --use_sumo
- For MARL: Use --marl flag with grid configurations.

Additional Dependencies:
- TensorFlow for forecasting
- Gymnasium for environments
