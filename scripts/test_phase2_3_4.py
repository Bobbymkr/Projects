"""
Test script for Phase 2.3 and 2.4.

Tests Distributional RL and Adversarial Training.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.traffic_env import TrafficEnv
from src.rl.distributional_rl import DistributionalDQNAgent, DistributionalRLConfig
from src.rl.adversarial_training import AdversarialTrainingWrapper, AdversarialConfig
from src.rl.training_stability import TrainingStabilityFramework
from src.rl.convergence_monitor import ConvergenceMonitor

def test_distributional_rl():
    """Test Phase 2.3: Distributional RL."""
    print("="*80)
    print("Testing Phase 2.3: Distributional RL")
    print("="*80)
    
    # Create environment
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "episode_horizon": 3600
    }
    env = TrafficEnv(config=config)
    
    # Test C51
    print(f"\n📊 Testing C51 Algorithm...")
    c51_config = DistributionalRLConfig(algorithm="C51", num_atoms=51)
    c51_agent = DistributionalDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=c51_config
    )
    
    obs, _ = env.reset()
    action = c51_agent.select_action(obs)
    print(f"   ✅ C51 agent created, action selected: {action}")
    
    # Test uncertainty estimation
    uncertainty = c51_agent.get_uncertainty(obs)
    print(f"   ✅ Uncertainty estimation: variance={uncertainty.get('variance', 0):.4f}")
    
    # Test QR-DQN
    print(f"\n📊 Testing QR-DQN Algorithm...")
    qr_config = DistributionalRLConfig(algorithm="QR-DQN", num_quantiles=200)
    qr_agent = DistributionalDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=qr_config
    )
    
    action = qr_agent.select_action(obs)
    print(f"   ✅ QR-DQN agent created, action selected: {action}")
    
    uncertainty = qr_agent.get_uncertainty(obs)
    print(f"   ✅ Uncertainty estimation: IQR={uncertainty.get('iqr', 0):.4f}")
    
    # Test IQN
    print(f"\n📊 Testing IQN Algorithm...")
    iqn_config = DistributionalRLConfig(algorithm="IQN")
    iqn_agent = DistributionalDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=iqn_config
    )
    
    action = iqn_agent.select_action(obs)
    print(f"   ✅ IQN agent created, action selected: {action}")
    
    uncertainty = iqn_agent.get_uncertainty(obs)
    print(f"   ✅ Uncertainty estimation: std={uncertainty.get('std', 0):.4f}")
    
    print(f"\n✅ Distributional RL Test Complete!")
    return True


def test_adversarial_training():
    """Test Phase 2.4: Adversarial Training."""
    print("\n" + "="*80)
    print("Testing Phase 2.4: Adversarial Training")
    print("="*80)
    
    # Create environment
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "episode_horizon": 3600
    }
    env = TrafficEnv(config=config)
    
    # Create adversarial wrapper
    adversarial_config = AdversarialConfig(
        enabled=True,
        adversarial_probability=0.3,
        noise_level=0.1,
        domain_randomization=True,
        worst_case_traffic=True
    )
    
    adv_env = AdversarialTrainingWrapper(env, adversarial_config)
    
    print(f"\n✅ Adversarial wrapper created")
    print(f"   Adversarial probability: {adversarial_config.adversarial_probability}")
    print(f"   Sensor noise level: {adversarial_config.noise_level}")
    print(f"   Domain randomization: {adversarial_config.domain_randomization}")
    
    # Test reset with adversarial
    obs, info = adv_env.reset()
    print(f"\n📊 Testing adversarial reset...")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Adversarial active: {info.get('adversarial', False)}")
    print(f"   Sensor noise level: {info.get('sensor_noise_level', 0)}")
    
    # Test step with adversarial
    print(f"\n📊 Testing adversarial step...")
    action = 0
    obs, reward, terminated, truncated, info = adv_env.step(action)
    print(f"   Reward: {reward:.4f}")
    print(f"   Adversarial: {info.get('adversarial', False)}")
    print(f"   Failed sensors: {info.get('failed_sensors', 0)}")
    
    # Test multiple steps
    adversarial_count = 0
    for i in range(10):
        obs, reward, terminated, truncated, info = adv_env.step(i % adv_env.action_space.n)
        if info.get('adversarial', False):
            adversarial_count += 1
        if terminated or truncated:
            obs, info = adv_env.reset()
    
    print(f"\n📊 Adversarial scenarios encountered: {adversarial_count}/10")
    
    print(f"\n✅ Adversarial Training Test Complete!")
    return True


def test_combined():
    """Test Phase 2.3 and 2.4 together."""
    print("\n" + "="*80)
    print("Testing Phase 2.3 + 2.4: Combined")
    print("="*80)
    
    # Create environment with adversarial wrapper
    config = {
        "num_lanes": 4,
        "phase_lanes": [[0, 1], [2, 3]],
        "min_green": 5,
        "max_green": 60,
        "green_step": 5,
        "cycle_yellow": 3,
        "cycle_all_red": 1,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40,
        "episode_horizon": 3600
    }
    env = TrafficEnv(config=config)
    
    adversarial_config = AdversarialConfig(
        enabled=True,
        adversarial_probability=0.2,
        noise_level=0.05,
        domain_randomization=True
    )
    adv_env = AdversarialTrainingWrapper(env, adversarial_config)
    
    # Create C51 agent
    dist_config = DistributionalRLConfig(algorithm="C51", num_atoms=51)
    agent = DistributionalDQNAgent(
        state_dim=adv_env.observation_space.shape[0],
        action_dim=adv_env.action_space.n,
        config=dist_config
    )
    
    # Create stability framework
    stability_framework = TrainingStabilityFramework(
        optimizer=agent.optimizer,
        policy_net=agent.policy_net,
        target_net=agent.target_net,
        config={"grad_clip_norm": 10.0}
    )
    
    print(f"\n✅ All components initialized")
    print(f"   Agent: C51 (Distributional RL)")
    print(f"   Environment: Adversarial Training Wrapper")
    
    # Training loop
    print(f"\n🎓 Training with Phase 2.3 + 2.4...")
    episode_rewards = []
    adversarial_episodes = 0
    
    for episode in range(50):
        obs, info = adv_env.reset()
        episode_reward = 0.0
        steps = 0
        
        if info.get('adversarial', False):
            adversarial_episodes += 1
        
        done = False
        while not done and steps < 100:  # Limit steps for testing
            # Select action
            epsilon = stability_framework.get_exploration_rate() if stability_framework else 0.0
            action = agent.select_action(obs, epsilon=epsilon)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = adv_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Store experience
            agent.push(obs, action, reward, next_obs, done)
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    stability_framework.clip_gradients()
                    stability_framework.update_target_network()
            
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        stability_framework.step_scheduler(metric=episode_reward)
        stability_framework.step_exploration()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/50 | Reward: {episode_reward:.2f} | "
                  f"Avg: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   Episodes: {len(episode_rewards)}")
    print(f"   Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Adversarial Episodes: {adversarial_episodes}/{len(episode_rewards)}")
    
    # Test uncertainty
    obs, _ = adv_env.reset()
    uncertainty = agent.get_uncertainty(obs)
    print(f"   Uncertainty (variance): {uncertainty.get('variance', 0):.4f}")
    
    print(f"\n✅ Combined Test Complete!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("Phase 2.3 & 2.4 Complete Test Suite")
    print("="*80)
    
    # Test Phase 2.3
    test_distributional_rl()
    
    # Test Phase 2.4
    test_adversarial_training()
    
    # Test Combined
    test_combined()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

