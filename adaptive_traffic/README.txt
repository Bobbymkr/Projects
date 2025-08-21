Adaptive Traffic Signal Control - Quickstart

This repo contains a minimal baseline to train a DQN agent to choose green time for a single intersection.

Folders:
- src/env: Gymnasium environment (queue-based)
- src/rl: DQN training and inference
- src/vision: Placeholder for YOLOv8-based queue extraction
- configs: configuration files

Steps:
1) Create venv: python -m venv .venv
2) Activate venv (Windows): .venv\\Scripts\\activate
3) Install deps: pip install -r requirements.txt
4) Train: python src\\rl\\train_dqn.py --episodes 5
5) Inference demo: python src\\rl\\inference.py

SUMO Integration:
- Install SUMO: pip install eclipse-sumo
- Ensure SUMO_HOME is set and binaries in PATH.
- Use --use_sumo flag for SUMO mode, e.g., python src\\rl\\train_dqn.py --use_sumo
