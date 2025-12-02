#!/usr/bin/env python3
"""
Quick Evaluation Script for 700+ Episode Trained Agent
This script provides a streamlined evaluation of your trained DQN agent.
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from stable_baselines3 import DQN
from src.env.traffic_env import TrafficEnv
from src.utils.config import load_config

def evaluate_trained_agent(model_path: str, config_path: str, num_episodes: int = 10):
    """
    Quick evaluation of your 700+ episode trained agent
    """
    print("🚦 EVALUATING YOUR 700+ EPISODE TRAINED AGENT")
    print("=" * 60)
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None
    
    # Load trained model
    try:
        model = DQN.load(model_path)
        print(f"✅ Trained model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create environment
    try:
        env = TrafficEnv(config)
        print(f"✅ Traffic environment created")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return None
    
    print(f"\n🔍 Running {num_episodes} evaluation episodes...")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    decision_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Count each action type
    total_decisions = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Handle tuple return from reset
        episode_reward = 0
        episode_length = 0
        episode_decisions = []
        
        print(f"  Episode {episode + 1}/{num_episodes}... ", end="", flush=True)
        
        done = False
        max_steps = 1000  # Limit episode length
        
        while not done and episode_length < max_steps:
            # Agent makes decision
            if isinstance(obs, tuple):
                obs = obs[0]  # Extract observation from tuple
            action = model.predict(obs, deterministic=True)[0]
            action = int(action)  # Ensure integer
            
            # Record decision
            episode_decisions.append(action)
            decision_counts[action] += 1
            total_decisions += 1
            
            # Take step
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # Calculate performance metrics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    # Decision analysis
    decision_percentages = {action: (count/total_decisions)*100 for action, count in decision_counts.items()}
    
    # Display Results
    print("\n" + "=" * 60)
    print("🎯 AGENT PERFORMANCE EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 REWARD PERFORMANCE:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Standard Deviation: {std_reward:.2f}")
    print(f"   Best Episode: {max(episode_rewards):.2f}")
    print(f"   Worst Episode: {min(episode_rewards):.2f}")
    
    print(f"\n⏱️  EPISODE STATISTICS:")
    print(f"   Average Episode Length: {avg_length:.1f} steps")
    print(f"   Total Decisions Made: {total_decisions}")
    
    print(f"\n🎮 DECISION ANALYSIS:")
    action_names = {0: "North-South Green", 1: "East-West Green", 2: "All Red", 3: "Smart Adaptive"}
    for action, percentage in decision_percentages.items():
        action_name = action_names.get(action, f"Action {action}")
        print(f"   {action_name}: {percentage:.1f}% ({decision_counts[action]} decisions)")
    
    # Performance Assessment
    print(f"\n🏆 PERFORMANCE ASSESSMENT:")
    
    # Reward-based assessment
    if avg_reward > -500:
        reward_grade = "A - Excellent"
    elif avg_reward > -1000:
        reward_grade = "B - Good"
    elif avg_reward > -1500:
        reward_grade = "C - Average"
    else:
        reward_grade = "D - Needs Improvement"
    
    print(f"   Reward Performance: {reward_grade}")
    
    # Decision diversity assessment
    non_zero_actions = sum(1 for count in decision_counts.values() if count > 0)
    if non_zero_actions >= 3:
        diversity_grade = "Good - Uses multiple strategies"
    elif non_zero_actions >= 2:
        diversity_grade = "Fair - Limited strategy diversity"
    else:
        diversity_grade = "Poor - Single strategy only"
    
    print(f"   Decision Diversity: {diversity_grade}")
    
    # Consistency assessment
    if std_reward < 200:
        consistency_grade = "High - Very consistent performance"
    elif std_reward < 400:
        consistency_grade = "Medium - Moderately consistent"
    else:
        consistency_grade = "Low - Inconsistent performance"
    
    print(f"   Performance Consistency: {consistency_grade}")
    
    # Accuracy estimate based on reward improvement
    baseline_reward = -2000  # Typical reward for random policy
    if avg_reward > baseline_reward:
        improvement = ((avg_reward - baseline_reward) / abs(baseline_reward)) * 100
        accuracy_estimate = max(0.0, min(100.0, 50.0 + float(improvement)))  # Scale to 0-100%
    else:
        accuracy_estimate = 25  # Below baseline
    
    print(f"\n🎯 ESTIMATED ACCURACY: {accuracy_estimate:.1f}%")
    print(f"   (Based on improvement over random baseline)")
    
    # Predictions about agent behavior
    print(f"\n🔮 AGENT BEHAVIOR PREDICTIONS:")
    dominant_action = max(decision_counts.keys(), key=lambda k: decision_counts[k])
    dominant_name = action_names.get(dominant_action, f"Action {dominant_action}")
    print(f"   Preferred Strategy: {dominant_name}")
    
    if decision_counts[3] > total_decisions * 0.3:
        print(f"   ✅ Agent uses adaptive control (good learning)")
    else:
        print(f"   ⚠️  Agent prefers fixed timing (may need more training)")
    
    if std_reward < 300:
        print(f"   ✅ Reliable performance across episodes")
    else:
        print(f"   ⚠️  Performance varies significantly between episodes")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes_evaluated": num_episodes,
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_episode_length": float(avg_length),
        "decision_distribution": decision_counts,
        "estimated_accuracy": float(accuracy_estimate),
        "performance_grade": reward_grade
    }
    
    results_file = "quick_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

def main():
    """Main evaluation function"""
    
    # Check for trained model
    model_path = "runs/dqn_traffic.zip"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("   Please ensure you have a trained model in the 'runs' directory")
        return
    
    print(f"🚀 Found trained model: {model_path}")
    
    # Configuration
    config_path = "configs/intersection.json"
    
    # Run evaluation
    results = evaluate_trained_agent(model_path, config_path, num_episodes=10)
    
    if results:
        print("\n🎉 Evaluation completed successfully!")
        print(f"Your 700+ episode agent shows {results['estimated_accuracy']:.1f}% accuracy!")
    else:
        print("\n❌ Evaluation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()