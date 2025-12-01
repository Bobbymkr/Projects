"""
Quick Start Example.

Simple example demonstrating basic usage of the Adaptive Traffic Control System.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.rl.dqn_agent import DQNAgent
from src.devtools.testing_utils import TrafficDataGenerator, TestEnvironment


def main():
    """Run quick start example."""
    print("🚦 Adaptive Traffic Control - Quick Start Example")
    print("=" * 60)
    
    # 1. Create traffic data generator
    print("\n1. Generating traffic data...")
    generator = TrafficDataGenerator(seed=42)
    traffic_state = generator.generate_state(num_lanes=4)
    print(f"   Generated state: {traffic_state.to_dict()}")
    
    # 2. Create test environment
    print("\n2. Creating test environment...")
    env = TestEnvironment(num_lanes=4, num_actions=4, max_steps=100)
    initial_state = env.reset()
    print(f"   Environment reset. Initial state ready.")
    
    # 3. Create DQN agent
    print("\n3. Creating DQN agent...")
    state_dim = len(initial_state.queue_lengths)
    action_dim = 4
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    print(f"   Agent created: state_dim={state_dim}, action_dim={action_dim}")
    
    # 4. Run a simple episode
    print("\n4. Running test episode...")
    state = env.reset()
    total_reward = 0.0
    steps = 0
    
    for step in range(10):  # Short episode for demo
        # Select action
        action = agent.select_action(state.queue_lengths, epsilon=1.0)  # Random for demo
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if done:
            break
        
        state = next_state
    
    print(f"   Episode complete: {steps} steps, total reward: {total_reward:.2f}")
    
    # 5. Demonstrate traffic scenario generation
    print("\n5. Generating rush hour scenario...")
    scenario = generator.generate_rush_hour_scenario(duration=10)
    print(f"   Generated {len(scenario)} traffic states")
    print(f"   Average queue length: {np.mean([s.queue_lengths.mean() for s in scenario]):.1f}")
    
    print("\n✅ Quick start example complete!")
    print("\nNext steps:")
    print("  - Read the documentation: docs/developer/GETTING_STARTED.md")
    print("  - Try the API examples: examples/api_example.py")
    print("  - Explore the research platform: examples/research_example.py")


if __name__ == "__main__":
    main()

