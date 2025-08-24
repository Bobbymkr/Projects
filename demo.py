#!/usr/bin/env python3
"""
Adaptive Traffic Signal Control - Demo Script

This script demonstrates the Adaptive Traffic Signal Control system by:
1. Training a DQN agent on a traffic intersection
2. Running inference to show optimal signal timing
3. Comparing with random actions
"""

import os
import sys
import subprocess
import time
import shutil
import argparse

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f" Command failed with return code {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(" Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f" Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Adaptive Traffic demo")
    parser.add_argument('--force-sumo', action='store_true', help='Force using SUMO even if not auto-detected')
    parser.add_argument('--no-sumo', action='store_true', help='Disable SUMO even if auto-detected')
    args = parser.parse_args()

    print("🚦 Adaptive Traffic Signal Control System")
    print("=" * 60)
    print("This demo will showcase the intelligent traffic signal control system")
    print("using Deep Q-Network (DQN) reinforcement learning.")
    print()
    
    # Check if virtual environment is activated
    if not os.path.exists(".venv"):
        print(" Virtual environment not found. Please run the setup first.")
        return

    # Detect SUMO availability and enable SUMO mode automatically if present
    def _sumo_available() -> bool:
        try:
            return shutil.which("sumo") is not None
        except Exception:
            return False

    detected_sumo = _sumo_available()
    # Command-line overrides
    if args.force_sumo:
        use_sumo = True
    elif args.no_sumo:
        use_sumo = False
    else:
        use_sumo = detected_sumo

    print(f" SUMO auto-detected: {detected_sumo}; using SUMO: {use_sumo}")

    # Ensure runs dir exists and set SUMO log path for SumoEnv
    runs_dir = os.path.join(os.getcwd(), 'runs')
    try:
        os.makedirs(runs_dir, exist_ok=True)
    except Exception:
        pass
    os.environ.setdefault('SUMO_LOG', os.path.join(runs_dir, 'sumo.log'))
    if use_sumo:
        print(f" Writing SUMO logs to: {os.environ.get('SUMO_LOG')}")
    
    # Step 1: Quick training
    print(" Step 1: Training the DQN Agent")
    print("Training a DQN agent on a 4-lane intersection for 5 episodes...")
    
    train_cmd = "python src/rl/train_dqn.py --episodes 5 --config configs/intersection.json"
    if use_sumo:
        train_cmd += " --use_sumo"

    success = run_command(
        train_cmd,
        "Training DQN Agent (5 episodes)"
    )
    
    if not success:
        print(" Training failed. Please check the error messages above.")
        return
    
    # Step 2: Run inference with trained model
    print("\n Step 2: Running Inference with Trained Model")
    print("The trained agent will now recommend optimal green light durations...")
    
    infer_cmd = "python src/rl/inference.py sim --model runs/dqn_traffic.zip"
    if use_sumo:
        infer_cmd += " --use_sumo"

    success = run_command(
        infer_cmd,
        "Running Inference with Trained Model"
    )
    
    if not success:
        print(" Inference failed. Please check the error messages above.")
        return
    
    # Step 3: Create visualization
    print("\n Step 3: Creating Visualization")
    print("Generating a plot showing queue dynamics over time...")
    
    success = run_command(
        "python src/rl/visualize_sim.py --model runs/dqn_traffic.zip --steps 15",
        "Creating Visualization with Trained Model"
    )
    
    if success:
        print("\n Demo completed successfully!")
        print("\n Generated files:")
        print("   - runs/dqn_traffic.zip (trained model)")
        print("   - runs/queue_timeseries.png (visualization)")
        print("   - runs/rewards.npy (training rewards)")
        print("\n The Adaptive Traffic Signal Control system is working!")
        print("\n Key Features Demonstrated:")
        print("   • Deep Q-Network (DQN) reinforcement learning")
        print("   • Real-time traffic signal optimization")
        print("   • Queue length monitoring and visualization")
        print("   • Adaptive response to traffic conditions")
    else:
        print(" Visualization failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
