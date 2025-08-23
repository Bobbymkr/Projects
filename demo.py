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
    print("🚦 Adaptive Traffic Signal Control System")
    print("=" * 60)
    print("This demo will showcase the intelligent traffic signal control system")
    print("using Deep Q-Network (DQN) reinforcement learning.")
    print()
    
    # Check if virtual environment is activated
    if not os.path.exists(".venv"):
        print(" Virtual environment not found. Please run the setup first.")
        return
    
    # Step 1: Quick training
    print(" Step 1: Training the DQN Agent")
    print("Training a DQN agent on a 4-lane intersection for 5 episodes...")
    
    success = run_command(
        "python src/rl/train_dqn.py --episodes 5 --config configs/intersection.json",
        "Training DQN Agent (5 episodes)"
    )
    
    if not success:
        print(" Training failed. Please check the error messages above.")
        return
    
    # Step 2: Run inference with trained model
    print("\n Step 2: Running Inference with Trained Model")
    print("The trained agent will now recommend optimal green light durations...")
    
    success = run_command(
        "python src/rl/inference.py sim --model runs/dqn_traffic.zip",
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
