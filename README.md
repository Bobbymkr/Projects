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


## Best Practices Followed

This project incorporates several best practices to ensure code quality, maintainability, and performance:

- **Modularity**: Code is structured into separate modules and classes (e.g., `MarlEnv`, `TrafficForecaster`) with extracted methods for validation and utilities.
- **Error Handling**: Implemented try-except blocks around critical operations like TraCI calls, model predictions, and file operations to enhance robustness.
- **Scalability**: Dynamic neighbor detection in MARL environments and parallel episode execution using `SubprocVecEnv` for efficient training.
- **Advanced Modeling**: Utilized a CNN-LSTM hybrid model for traffic forecasting to capture spatial and temporal dependencies effectively.
- **Logging and Monitoring**: Integrated TensorBoard for visualizing training metrics and progress.
- **Documentation**: Added detailed function-level comments explaining parameters, returns, and logic in key files like `marl_env.py`, `traffic_forecast.py`, and `train_dqn.py`.
- **Hyperparameter Tuning**: Employed Optuna for optimizing DQN parameters.

These practices follow industry standards for RL and simulation projects.
