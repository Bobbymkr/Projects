#!/usr/bin/env python3
"""
Simplified Adaptive Traffic Demo - Works with minimal dependencies
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_traffic_demo():
    """Run a simple traffic control demonstration"""
    print("🚦 Adaptive Traffic Signal Control System - Simplified Demo")
    print("=" * 70)
    
    try:
        from env.traffic_env import TrafficEnv
        
        # Create traffic environment
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 10,
            "max_green": 30,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 2,
            "queue_capacity": 50,
            "arrival_rates": [0.4, 0.4, 0.3, 0.3],
            "episode_horizon": 600,
            "reward_weights": {"queue": -1.0, "wait_penalty": -0.1}
        }
        
        env = TrafficEnv(config)
        print(f"✅ Created 4-lane intersection environment")
        print(f"   - Green time range: {config['min_green']}-{config['max_green']} seconds")
        print(f"   - Arrival rates: {config['arrival_rates']} vehicles/sec per lane")
        print(f"   - Queue capacity: {config['queue_capacity']} vehicles per lane")
        
        # Simulate different control strategies
        strategies = {
            "Random Policy": lambda env: np.random.randint(0, env.action_space.n),
            "Short Green Policy": lambda env: 0,  # Always minimum green
            "Long Green Policy": lambda env: env.action_space.n - 1,  # Always maximum green
            "Adaptive Policy": adaptive_policy  # Simple adaptive strategy
        }
        
        results = {}
        
        for strategy_name, policy in strategies.items():
            print(f"\n🔄 Testing {strategy_name}")
            print("-" * 50)
            
            # Reset environment
            obs, info = env.reset(seed=42)
            total_reward = 0
            queue_history = []
            processed_vehicles = []
            step_count = 0
            
            # Run for 30 steps (about 10 minutes simulation time)
            for step in range(30):
                action = policy(env)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                queue_history.append(np.sum(info['queues']))
                processed_vehicles.append(info['total_vehicles_processed'])
                step_count += 1
                
                if step % 10 == 0:
                    print(f"  Step {step+1:2d}: Total Queue = {np.sum(info['queues']):2d}, "
                          f"Processed = {info['total_vehicles_processed']:3d}, "
                          f"Reward = {reward:6.2f}")
                
                if terminated or truncated:
                    break
            
            # Store results
            results[strategy_name] = {
                'total_reward': total_reward,
                'avg_queue': np.mean(queue_history),
                'final_processed': processed_vehicles[-1] if processed_vehicles else 0,
                'queue_history': queue_history,
                'processed_history': processed_vehicles
            }
            
            print(f"  ✅ {strategy_name} Results:")
            print(f"     Total Reward: {total_reward:.2f}")
            print(f"     Average Queue: {np.mean(queue_history):.1f} vehicles")
            print(f"     Vehicles Processed: {processed_vehicles[-1] if processed_vehicles else 0}")
        
        # Create visualization
        create_comparison_plots(results)
        
        # Print summary
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Total Reward':<15} {'Avg Queue':<12} {'Processed':<10}")
        print("-" * 70)
        
        best_strategy = ""
        best_reward = float('-inf')
        
        for name, result in results.items():
            print(f"{name:<20} {result['total_reward']:<15.2f} "
                  f"{result['avg_queue']:<12.1f} {result['final_processed']:<10d}")
            
            if result['total_reward'] > best_reward:
                best_reward = result['total_reward']
                best_strategy = name
        
        print(f"\n🏆 Best Strategy: {best_strategy} (Reward: {best_reward:.2f})")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\nGenerated files:")
        if os.path.exists('traffic_comparison.png'):
            print("  - traffic_comparison.png (strategy comparison)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def adaptive_policy(env):
    """Simple adaptive policy based on queue lengths"""
    # Get current observation (normalized queue lengths)
    if hasattr(env, 'queues'):
        current_queues = env.queues
        total_queue = np.sum(current_queues)
        
        # Adaptive logic:
        # - If queues are long, use longer green times
        # - If queues are short, use shorter green times
        if total_queue > 30:
            return env.action_space.n - 1  # Maximum green
        elif total_queue > 15:
            return env.action_space.n // 2  # Medium green
        else:
            return 0  # Minimum green
    else:
        return env.action_space.n // 2  # Default to medium

def create_comparison_plots(results):
    """Create visualization comparing different strategies"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Queue lengths over time
        ax1.set_title('Queue Length Evolution by Strategy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Queue Length')
        ax1.grid(True, alpha=0.3)
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (name, result) in enumerate(results.items()):
            ax1.plot(result['queue_history'], label=name, 
                    color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
        ax1.legend()
        
        # Plot 2: Vehicles processed over time
        ax2.set_title('Cumulative Vehicles Processed by Strategy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Vehicles Processed')
        ax2.grid(True, alpha=0.3)
        
        for i, (name, result) in enumerate(results.items()):
            ax2.plot(result['processed_history'], label=name, 
                    color=colors[i % len(colors)], linewidth=2, marker='s', markersize=4)
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('traffic_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Created traffic_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    success = run_traffic_demo()
    if success:
        print("\n🚀 Next Steps:")
        print("  - Explore the full demo with: python demo.py")
        print("  - Run tests with: python -m pytest tests/")
        print("  - Train a DQN agent with: python src/rl/train_dqn.py")
    else:
        print("\n⚠️  Some issues occurred. Please check dependencies.")#!/usr/bin/env python3
"""
Simplified Adaptive Traffic Demo - Works with minimal dependencies
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_traffic_demo():
    """Run a simple traffic control demonstration"""
    print("🚦 Adaptive Traffic Signal Control System - Simplified Demo")
    print("=" * 70)
    
    try:
        from env.traffic_env import TrafficEnv
        
        # Create traffic environment
        config = {
            "num_lanes": 4,
            "phase_lanes": [[0, 1], [2, 3]],
            "min_green": 10,
            "max_green": 30,
            "green_step": 5,
            "cycle_yellow": 3,
            "cycle_all_red": 2,
            "queue_capacity": 50,
            "arrival_rates": [0.4, 0.4, 0.3, 0.3],
            "episode_horizon": 600,
            "reward_weights": {"queue": -1.0, "wait_penalty": -0.1}
        }
        
        env = TrafficEnv(config)
        print(f"✅ Created 4-lane intersection environment")
        print(f"   - Green time range: {config['min_green']}-{config['max_green']} seconds")
        print(f"   - Arrival rates: {config['arrival_rates']} vehicles/sec per lane")
        print(f"   - Queue capacity: {config['queue_capacity']} vehicles per lane")
        
        # Simulate different control strategies
        strategies = {
            "Random Policy": lambda env: np.random.randint(0, env.action_space.n),
            "Short Green Policy": lambda env: 0,  # Always minimum green
            "Long Green Policy": lambda env: env.action_space.n - 1,  # Always maximum green
            "Adaptive Policy": adaptive_policy  # Simple adaptive strategy
        }
        
        results = {}
        
        for strategy_name, policy in strategies.items():
            print(f"\n🔄 Testing {strategy_name}")
            print("-" * 50)
            
            # Reset environment
            obs, info = env.reset(seed=42)
            total_reward = 0
            queue_history = []
            processed_vehicles = []
            step_count = 0
            
            # Run for 30 steps (about 10 minutes simulation time)
            for step in range(30):
                action = policy(env)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                queue_history.append(np.sum(info['queues']))
                processed_vehicles.append(info['total_vehicles_processed'])
                step_count += 1
                
                if step % 10 == 0:
                    print(f"  Step {step+1:2d}: Total Queue = {np.sum(info['queues']):2d}, "
                          f"Processed = {info['total_vehicles_processed']:3d}, "
                          f"Reward = {reward:6.2f}")
                
                if terminated or truncated:
                    break
            
            # Store results
            results[strategy_name] = {
                'total_reward': total_reward,
                'avg_queue': np.mean(queue_history),
                'final_processed': processed_vehicles[-1] if processed_vehicles else 0,
                'queue_history': queue_history,
                'processed_history': processed_vehicles
            }
            
            print(f"  ✅ {strategy_name} Results:")
            print(f"     Total Reward: {total_reward:.2f}")
            print(f"     Average Queue: {np.mean(queue_history):.1f} vehicles")
            print(f"     Vehicles Processed: {processed_vehicles[-1] if processed_vehicles else 0}")
        
        # Create visualization
        create_comparison_plots(results)
        
        # Print summary
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Total Reward':<15} {'Avg Queue':<12} {'Processed':<10}")
        print("-" * 70)
        
        best_strategy = ""
        best_reward = float('-inf')
        
        for name, result in results.items():
            print(f"{name:<20} {result['total_reward']:<15.2f} "
                  f"{result['avg_queue']:<12.1f} {result['final_processed']:<10d}")
            
            if result['total_reward'] > best_reward:
                best_reward = result['total_reward']
                best_strategy = name
        
        print(f"\n🏆 Best Strategy: {best_strategy} (Reward: {best_reward:.2f})")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\nGenerated files:")
        if os.path.exists('traffic_comparison.png'):
            print("  - traffic_comparison.png (strategy comparison)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def adaptive_policy(env):
    """Simple adaptive policy based on queue lengths"""
    # Get current observation (normalized queue lengths)
    if hasattr(env, 'queues'):
        current_queues = env.queues
        total_queue = np.sum(current_queues)
        
        # Adaptive logic:
        # - If queues are long, use longer green times
        # - If queues are short, use shorter green times
        if total_queue > 30:
            return env.action_space.n - 1  # Maximum green
        elif total_queue > 15:
            return env.action_space.n // 2  # Medium green
        else:
            return 0  # Minimum green
    else:
        return env.action_space.n // 2  # Default to medium

def create_comparison_plots(results):
    """Create visualization comparing different strategies"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Queue lengths over time
        ax1.set_title('Queue Length Evolution by Strategy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Queue Length')
        ax1.grid(True, alpha=0.3)
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (name, result) in enumerate(results.items()):
            ax1.plot(result['queue_history'], label=name, 
                    color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
        ax1.legend()
        
        # Plot 2: Vehicles processed over time
        ax2.set_title('Cumulative Vehicles Processed by Strategy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Vehicles Processed')
        ax2.grid(True, alpha=0.3)
        
        for i, (name, result) in enumerate(results.items()):
            ax2.plot(result['processed_history'], label=name, 
                    color=colors[i % len(colors)], linewidth=2, marker='s', markersize=4)
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('traffic_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Created traffic_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    success = run_traffic_demo()
    if success:
        print("\n🚀 Next Steps:")
        print("  - Explore the full demo with: python demo.py")
        print("  - Run tests with: python -m pytest tests/")
        print("  - Train a DQN agent with: python src/rl/train_dqn.py")
    else:
        print("\n⚠️  Some issues occurred. Please check dependencies.")