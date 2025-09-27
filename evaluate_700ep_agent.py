#!/usr/bin/env python3
"""
Comprehensive Evaluation of 700+ Episode Trained DQN Agent

This script evaluates the performance and accuracy of your trained agent
across multiple scenarios and provides detailed accuracy metrics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.env.traffic_env import TrafficEnv
from stable_baselines3 import DQN
from src.rl.dqn_agent import DQNAgent, DQNConfig

def load_config(path: str):
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_agent_performance(model_path: str, config_path: str, episodes: int = 50):
    """
    Comprehensive evaluation of trained DQN agent.
    
    Returns:
        dict: Performance metrics including accuracy, consistency, and efficiency
    """
    print(f"🚦 EVALUATING 700+ EPISODE TRAINED AGENT")
    print("=" * 60)
    
    # Load configuration and create environment
    config = load_config(config_path)
    env = TrafficEnv(config)
    
    # Load trained model
    try:
        model = DQN.load(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    queue_management = []
    decision_consistency = []
    response_times = []
    
    # Traffic scenarios for testing
    scenarios = [
        ("Light Traffic", [0.2, 0.15, 0.25, 0.1]),
        ("Moderate Traffic", [0.3, 0.25, 0.35, 0.2]),
        ("Heavy Traffic", [0.5, 0.4, 0.6, 0.45]),
        ("Rush Hour", [0.7, 0.6, 0.8, 0.65])
    ]
    
    scenario_results = {}
    
    for scenario_name, arrival_rates in scenarios:
        print(f"\n📊 Testing Scenario: {scenario_name}")
        print(f"   Arrival rates: {arrival_rates}")
        
        # Update environment config for this scenario
        config["arrival_rates"] = arrival_rates
        test_env = TrafficEnv(config)
        
        scenario_rewards = []
        scenario_queues = []
        scenario_decisions = []
        
        for episode in range(episodes // len(scenarios)):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_queue_lengths = []
            episode_actions = []
            
            terminated = truncated = False
            
            while not (terminated or truncated):
                # Measure decision time
                start_time = time.time()
                action, _ = model.predict(obs, deterministic=True)
                decision_time = time.time() - start_time
                response_times.append(decision_time)
                
                # Execute action
                next_obs, reward, terminated, truncated, step_info = test_env.step(int(action))
                
                # Record metrics
                episode_reward += reward
                episode_steps += 1
                episode_queue_lengths.append(np.sum(step_info['queues']))
                episode_actions.append(action)
                
                obs = next_obs
            
            # Store episode results
            scenario_rewards.append(episode_reward)
            scenario_queues.append(np.mean(episode_queue_lengths))
            scenario_decisions.append(episode_actions)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            queue_management.append(np.mean(episode_queue_lengths))
        
        # Calculate scenario-specific metrics
        scenario_results[scenario_name] = {
            "avg_reward": np.mean(scenario_rewards),
            "reward_std": np.std(scenario_rewards),
            "avg_queue_length": np.mean(scenario_queues),
            "queue_std": np.std(scenario_queues),
            "decision_variance": np.var([len(set([int(d) for d in decisions])) for decisions in scenario_decisions])
        }
        
        print(f"   ✅ Avg Reward: {np.mean(scenario_rewards):.2f} ± {np.std(scenario_rewards):.2f}")
        print(f"   📊 Avg Queue Length: {np.mean(scenario_queues):.1f} ± {np.std(scenario_queues):.1f}")
    
    # Calculate overall performance metrics
    performance_metrics = {
        "overall_performance": {
            "avg_reward": np.mean(episode_rewards),
            "reward_std": np.std(episode_rewards),
            "best_episode_reward": np.max(episode_rewards),
            "worst_episode_reward": np.min(episode_rewards),
            "reward_consistency": 1 - (np.std(episode_rewards) / abs(np.mean(episode_rewards)))
        },
        "efficiency_metrics": {
            "avg_episode_length": np.mean(episode_lengths),
            "avg_queue_management": np.mean(queue_management),
            "avg_response_time_ms": np.mean(response_times) * 1000,
            "response_time_std_ms": np.std(response_times) * 1000
        },
        "scenario_results": scenario_results
    }
    
    return performance_metrics

def calculate_accuracy_metrics(performance_data):
    """Calculate accuracy and prediction quality metrics."""
    
    print(f"\n🎯 ACCURACY & PREDICTION QUALITY ANALYSIS")
    print("=" * 60)
    
    # Decision accuracy (consistency and appropriateness)
    reward_consistency = performance_data["overall_performance"]["reward_consistency"]
    avg_reward = performance_data["overall_performance"]["avg_reward"]
    
    # Baseline comparison (random policy would get approximately -300 to -500)
    baseline_reward = -400  # Estimated random policy performance
    improvement_over_baseline = ((avg_reward - baseline_reward) / abs(baseline_reward)) * 100
    
    # Response time accuracy (real-time capability)
    avg_response_time = performance_data["efficiency_metrics"]["avg_response_time_ms"]
    real_time_accuracy = 100 if avg_response_time < 100 else max(0, 100 - (avg_response_time - 100))
    
    # Queue management effectiveness
    avg_queue = performance_data["efficiency_metrics"]["avg_queue_management"]
    max_capacity = 40  # From config
    queue_efficiency = max(0, 100 - (avg_queue / max_capacity * 100))
    
    # Scenario adaptability (how well it adapts to different traffic conditions)
    scenario_rewards = [data["avg_reward"] for data in performance_data["scenario_results"].values()]
    adaptability = 100 - (np.std(scenario_rewards) / abs(np.mean(scenario_rewards)) * 100)
    
    accuracy_metrics = {
        "decision_consistency": reward_consistency * 100,
        "improvement_over_baseline": improvement_over_baseline,
        "real_time_accuracy": real_time_accuracy,
        "queue_management_efficiency": queue_efficiency,
        "scenario_adaptability": max(0.0, float(adaptability)),
        "overall_accuracy_score": np.mean([
            reward_consistency * 100,
            min(100, max(0, improvement_over_baseline)),
            real_time_accuracy,
            queue_efficiency,
            max(0.0, float(adaptability))
        ])
    }
    
    return accuracy_metrics

def display_results(performance_data, accuracy_data):
    """Display comprehensive evaluation results."""
    
    print(f"\n🏆 FINAL EVALUATION RESULTS (700+ Episodes)")
    print("=" * 60)
    
    # Overall Performance
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Average Reward: {performance_data['overall_performance']['avg_reward']:.2f}")
    print(f"   Best Episode: {performance_data['overall_performance']['best_episode_reward']:.2f}")
    print(f"   Consistency: {performance_data['overall_performance']['reward_consistency']*100:.1f}%")
    
    # Efficiency Metrics
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"   Response Time: {performance_data['efficiency_metrics']['avg_response_time_ms']:.2f} ms")
    print(f"   Queue Management: {performance_data['efficiency_metrics']['avg_queue_management']:.1f} avg vehicles")
    print(f"   Episode Length: {performance_data['efficiency_metrics']['avg_episode_length']:.0f} steps")
    
    # Accuracy Analysis
    print(f"\n🎯 ACCURACY ANALYSIS:")
    print(f"   Overall Accuracy Score: {accuracy_data['overall_accuracy_score']:.1f}%")
    print(f"   Decision Consistency: {accuracy_data['decision_consistency']:.1f}%")
    print(f"   Improvement over Baseline: {accuracy_data['improvement_over_baseline']:+.1f}%")
    print(f"   Real-time Capability: {accuracy_data['real_time_accuracy']:.1f}%")
    print(f"   Queue Management Efficiency: {accuracy_data['queue_management_efficiency']:.1f}%")
    print(f"   Scenario Adaptability: {accuracy_data['scenario_adaptability']:.1f}%")
    
    # Scenario Performance
    print(f"\n🚦 SCENARIO-SPECIFIC PERFORMANCE:")
    for scenario, data in performance_data['scenario_results'].items():
        print(f"   {scenario}:")
        print(f"     Reward: {data['avg_reward']:.2f} ± {data['reward_std']:.2f}")
        print(f"     Queue Length: {data['avg_queue_length']:.1f} ± {data['queue_std']:.1f}")
    
    # Performance Grade
    accuracy_score = accuracy_data['overall_accuracy_score']
    if accuracy_score >= 90:
        grade = "A+ (Excellent)"
    elif accuracy_score >= 80:
        grade = "A (Very Good)"
    elif accuracy_score >= 70:
        grade = "B+ (Good)"
    elif accuracy_score >= 60:
        grade = "B (Fair)"
    else:
        grade = "C (Needs Improvement)"
    
    print(f"\n🏅 PERFORMANCE GRADE: {grade}")
    print(f"📊 ACCURACY SCORE: {accuracy_score:.1f}/100")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if accuracy_score >= 80:
        print("   ✅ Agent is performing excellently! Ready for production deployment.")
        print("   ✅ Consider testing on real-world scenarios or larger networks.")
    elif accuracy_score >= 70:
        print("   📈 Good performance! Consider fine-tuning for specific scenarios.")
        print("   🔧 Monitor performance in production and collect more data.")
    else:
        print("   🔄 Consider additional training or hyperparameter optimization.")
        print("   📊 Analyze decision patterns for potential improvements.")

def main():
    """Main evaluation function."""
    # Check for trained model
    model_paths = [
        "runs/dqn_traffic.zip",
        "runs/extended_training_500ep/dqn_model.zip",
        "runs/checkpoint/dqn_model.zip"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found! Please train an agent first.")
        return
    
    print(f"🚀 Found trained model: {model_path}")
    
    # Run evaluation
    config_path = "configs/intersection.json"
    performance_data = evaluate_agent_performance(model_path, config_path, episodes=40)
    
    if performance_data is None:
        return
    
    # Calculate accuracy metrics
    accuracy_data = calculate_accuracy_metrics(performance_data)
    
    # Display results
    display_results(performance_data, accuracy_data)
    
    # Save results
    results = {
        "performance_data": performance_data,
        "accuracy_data": accuracy_data,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path
    }
    
    with open("runs/700ep_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to: runs/700ep_evaluation_results.json")

if __name__ == "__main__":
    main()