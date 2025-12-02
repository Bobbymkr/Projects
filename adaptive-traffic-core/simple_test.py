#!/usr/bin/env python3
"""
Simple test to run the Adaptive Traffic Signal Control System
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_environment():
    """Test the basic traffic environment"""
    print("🚦 Testing Adaptive Traffic Environment")
    print("=" * 50)
    
    try:
        from env.traffic_env import TrafficEnv
        
        # Simple configuration
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 5,
            "max_green": 60,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 1,
            "queue_capacity": 40,
            "arrival_rates": [0.3, 0.3, 0.2, 0.2],
            "episode_horizon": 300,
        }
        
        # Create environment
        env = TrafficEnv(config)
        print(f"✅ Created traffic environment with {config['num_lanes']} lanes")
        
        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset. Initial queues: {info['queues']}")
        
        # Run a few steps
        total_reward = 0
        queue_history = []
        
        for step in range(10):
            # Random action for testing
            action = np.random.randint(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            queue_history.append(info['queues'].copy())
            
            print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
                  f"Queues={info['queues']}, Time={info['time']}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Completed {step+1} steps. Total reward: {total_reward:.2f}")
        
        # Create simple visualization
        if len(queue_history) > 1:
            try:
                queue_array = np.array(queue_history)
                plt.figure(figsize=(10, 6))
                for lane in range(config['num_lanes']):
                    plt.plot(queue_array[:, lane], label=f'Lane {lane}', marker='o')
                
                plt.xlabel('Time Step')
                plt.ylabel('Queue Length')
                plt.title('Traffic Queue Evolution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                output_file = 'simple_traffic_test.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved visualization to {output_file}")
                
            except Exception as e:
                print(f"⚠️  Could not create visualization: {e}")
        
        print("\n🎉 Basic traffic environment test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Missing dependencies. Please install:")
        print("  pip install numpy matplotlib gymnasium")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def test_simple_rl():
    """Test simple random policy vs environment"""
    print("\n🤖 Testing Simple RL Interaction")
    print("=" * 50)
    
    try:
        from env.traffic_env import TrafficEnv
        
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 10,
            "max_green": 30,
            "green_step": 5,
            "episode_horizon": 180,
            "arrival_rates": [0.4, 0.4, 0.3, 0.3]
        }
        
        env = TrafficEnv(config)
        
        # Test random policy
        print("Testing random policy...")
        obs, _ = env.reset(seed=123)
        total_reward = 0
        
        for step in range(20):
            action = np.random.randint(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"  Step {step}: Total queue = {np.sum(info['queues'])}, "
                      f"Cumulative reward = {total_reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Random policy completed. Final reward: {total_reward:.2f}")
        
        # Test fixed policy (always use medium green time)
        print("\nTesting fixed medium green policy...")
        obs, _ = env.reset(seed=123)
        total_reward_fixed = 0
        
        for step in range(20):
            action = len(env.green_values) // 2  # Middle action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward_fixed += reward
            
            if step % 5 == 0:
                print(f"  Step {step}: Total queue = {np.sum(info['queues'])}, "
                      f"Cumulative reward = {total_reward_fixed:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Fixed policy completed. Final reward: {total_reward_fixed:.2f}")
        
        # Compare policies
        print(f"\n📊 Policy Comparison:")
        print(f"  Random policy reward: {total_reward:.2f}")
        print(f"  Fixed policy reward:  {total_reward_fixed:.2f}")
        
        if total_reward_fixed > total_reward:
            print("✅ Fixed policy performed better!")
        else:
            print("✅ Random policy performed better (or same)!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in RL test: {e}")
        return False

if __name__ == "__main__":
    print("🚦 Adaptive Traffic Signal Control - Quick Test")
    print("=" * 60)
    
    success1 = test_basic_environment()
    success2 = test_simple_rl()
    
    if success1 and success2:
        print("\n🎉 All tests completed successfully!")
        print("\nThe Adaptive Traffic Signal Control system is working!")
        print("\nGenerated files:")
        if os.path.exists('simple_traffic_test.png'):
            print("  - simple_traffic_test.png (traffic visualization)")
        
        print("\nNext steps:")
        print("  - Run 'python demo.py' for the full demonstration")
        print("  - Explore the RL training with 'python src/rl/train_dqn.py'")
        print("  - Check the testing with 'python -m pytest tests/'")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")